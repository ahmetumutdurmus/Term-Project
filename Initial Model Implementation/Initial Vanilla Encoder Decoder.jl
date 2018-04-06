using Knet

function datainit(vocabSize, sentenceLength)
     info("Program has started.")
     #filenames = ["dev08-11"]
     filenames = ["dev08-11"]#, "nc9"]
     devfiles = ["ntst1213"]
     testfiles = ["ntst14"]

     FrTrn = ""
     for filename in filenames
          strings = open(filename * ".fr", "r") do f
               return readstring(f)
          end
          FrTrn = FrTrn * strings
     end
     FrTrn = split(FrTrn)

     EnTrn = ""
     for filename in filenames
          strings = open(filename * ".en", "r") do f
               return readstring(f)
          end
          EnTrn = EnTrn * strings
     end
     EnTrn = split(EnTrn)
     info("Tokenized data is loaded.")
     function Coverage(tokens, dictSize)
          function CountingOccurences(tokens)
               vocab = Dict{String,Int}()
               for f in tokens
                    vocab[f] = get(vocab,f,0) + 1
               end
               return vocab
          end
          num = dictSize - 2
          vocab = CountingOccurences(tokens)
          if length(vocab) > num
               vocab = Dict(sort(collect(vocab), by = tuple -> last(tuple), rev=true)[1:num])
          end

          for (i,key) in enumerate(collect(keys(vocab)))
               vocab[key] = i
          end
          vocab["[UNK]"] = num + 1
          vocab["</s>"] = num + 2
          n = 0
          for f in tokens
               if haskey(vocab, f)
                    n += 1
               end
          end
          percent = n / length(tokens)
          return vocab, percent
     end

     EnVocab, = Coverage(EnTrn, vocabSize)
     FrVocab, = Coverage(FrTrn, vocabSize)
     info("Vocabularies are constructed.")
     function datagetter(filesnames)
          EnLine = String[]
          for filename in filesnames
               strings = open(filename * ".en", "r") do f
                    return readlines(f)
               end
               EnLine = cat(1, EnLine, strings)
          end


          FrLine = String[]
          for filename in filesnames
               strings = open(filename * ".fr", "r") do f
                    return readlines(f)
               end
               FrLine = cat(1, FrLine, strings)
          end

          EnVec = Array{Array{Int}}(length(EnLine))
          for (i,line) in enumerate(EnLine)
               EnVec[i] = vcat(map(x -> get(EnVocab, x, vocabSize-1), split(line)), vocabSize)
          end

          FrVec = Array{Array{Int}}(length(FrLine))
          for (i,line) in enumerate(FrLine)
               FrVec[i] = vcat(map(x -> get(FrVocab, x, vocabSize-1), split(line)), vocabSize)
          end
          info("Unsorted data is initialized.")
          data = Any[]
          for i = 1:length(EnVec)
               if length(EnVec[i]) <= sentenceLength && length(FrVec[i]) <= sentenceLength
                    push!(data, (EnVec[i], FrVec[i]))
               end
          end
          sort!(data, by = x->length(x[2]), rev = true)
          sort!(data, by = x->length(x[1]), rev = true)
          return data
     end
     trn = datagetter(filenames)
     dev = datagetter(devfiles)
     tst = datagetter(testfiles)
     return trn, dev, tst, EnVocab, FrVocab
end

function minibatch(data, batchsize)
     output = Any[]
     for i = 1:batchsize:length(data)
          bl = min(i+batchsize-1, length(data))
          EnData = Any[]
          FrData = Any[]
          for j = i:bl
               Eni, Fri = data[j]
               push!(EnData, Eni)
               push!(FrData, Fri)
          end
          EnConcat, EnBatch = preRNN(EnData)
          FrOrder = sortperm(FrData, by = length, rev = true)
          FrData = FrData[FrOrder]
          FrConcat, FrBatch = preRNN(FrData)
          push!(output,(EnConcat, EnBatch, FrConcat, FrBatch, FrOrder))
     end
     info("Minibatches are created.")
     return output
end

function preRNN(input)
     lenvec = length.(input)
     longest = length(input[1])
     batchsizes = zeros(Int, longest)
     output = Int[]
     for i=1:longest
          batchsizes[i] = sum(i .<= lenvec)
          inputi = zeros(Int, batchsizes[i])
          for j = 1:batchsizes[i]
               inputi[j] = input[j][i]
          end
     output = vcat(output, inputi)
     end
     return Int.(output), Int.(batchsizes)
end

function modelinit(inputSize, embeddingSize, hiddenSize, maxoutSize)
     winit = 0.1
     w = Any[]
     rx, wrx = rnninit(embeddingSize, hiddenSize, rnnType = :gru, bidirectional = false)
     push!(w, rx) #Encoder struct
     push!(w, wrx) #Encoder rnn weights
     push!(w, randn(embeddingSize, inputSize) * winit) #Encoder embedding

     push!(w, randn(hiddenSize, hiddenSize) * winit) #hidden to initial annotation
     push!(w, randn(embeddingSize, inputSize) * winit) #Decoder embedding
     push!(w, randn(maxoutSize * 2, hiddenSize) * winit) #tiTilda
     push!(w, randn(maxoutSize * 2, embeddingSize + hiddenSize) * winit) #tiTilda
     push!(w, randn(inputSize, maxoutSize) * winit)
     ry, wry = rnninit(embeddingSize + hiddenSize, hiddenSize, rnnType = :gru)
     push!(w, wry)
     push!(w, ry)

     if gpu() >= 0
          for i = 3:length(w) - 2
               w[i] = convert(KnetArray{Float32},w[i])
          end
     end
     if gpu() == -1
          for i = 3:length(w) - 2
               w[i] = convert(Array{Float32},w[i])
          end
     end
     return w
end

function predict(w, x, xbatchsizes, y, ybatchsizes, yorder)
     inithiddens, annotations = encoder(w, x, xbatchsizes, yorder)
     ttilda = decoder(w, y, ybatchsizes, annotations, inithiddens)
     scores = w[8] * ttilda
     return scores
end

function encoder(w, x, xbatchsizes, yorder)
     input = w[3][:,x]
     hiddens, lasthiddens,  = rnnforw(w[1], w[2], input, batchSizes = xbatchsizes, hy = true)
     inithiddens = hiddens[:, yorder]
     lasthiddens = lasthiddens[:, :, 1]
     lasthiddens = lasthiddens[:, yorder]
     return inithiddens, lasthiddens
end

function maxout(input)
     len = size(input,2)
     return reshape(maximum(reshape(input, 2, :), 1), :, len)
end

function decoder(w, y, batchsizes, annotations, inithiddens)
     yi = size(w[3],2) * ones(Int64, batchsizes[1])
     s = tanh.(w[4] * inithiddens)
     h = annotations
     sofar = 0
     ttilda = Any[]
     for i = 1:length(batchsizes)
          bsize = batchsizes[i]
          input = vcat(w[5][:,yi], h)
          titilda = w[6] * s + w[7] * input
          ti = maxout(titilda)
          push!(ttilda,ti[:,1:bsize])
          s = rnnforw(w[end], w[end-1], input, s)[1]
          s = s[:,1:bsize]
          yi = y[sofar + 1 : sofar + bsize]
          sofar += bsize
          h = annotations[:, 1:bsize]
     end
     return hcat(ttilda...)
end

function loss(w, data)
     x, xbatchsizes, y, ybatchsizes, yorder = data
     scores = predict(w, x, xbatchsizes, y, ybatchsizes, yorder)
     nll(scores, y, average = false)
end

lossgradient = gradloss(loss)

function train(w, data)
     totalloss = 0.0
     sentencecount = 0
     opt = optimizers(w, Adadelta)
     for batch in data
          gradients, lossval = lossgradient(w, batch)
          update!(w, gradients, opt)
          totalloss += lossval
          sentencecount += batch[2][1]
     end
     return totalloss / sentencecount
end

function reArrange(data, batchsizes)
     batchsize = batchsizes[1]
     output = Any[]
     for i=1:batchsize
          yi = zeros(Int64,sum(i .<= batchsizes))
          counter = 0
          for j = 1:length(yi)
               yi[j] = data[i + counter]
               counter += batchsizes[j]
          end
          push!(output, yi)
     end
     return output
end

function nGramCounts(source, target, n)
     len = length(source)
     sourcestring = string.(source)
     targetstring = string.(target)
     if len < n
          return 0, 0
     end
     nGramSource = Array{String}(len - n + 1)
     nGramTarget = Array{String}(len - n + 1)
     matchedCount = 0
     totalCount = 0
     for i = 1:len - n + 1
          nGramSource[i] = ""
          nGramTarget[i] = ""
          for j = i: i + n - 1
               nGramSource[i] = nGramSource[i] * sourcestring[j] * " "
               nGramTarget[i] = nGramTarget[i] * targetstring[j] * " "
          end
     end
     for i = 1:len - n + 1
          sourceoccurence = sum(nGramSource[i] .== nGramSource)
          targetoccurence = sum(nGramSource[i] .== nGramTarget)
          matchedCount += min(sourceoccurence, targetoccurence) / sourceoccurence
          totalCount += 1
     end
     return matchedCount, totalCount
end

function bleuCalc(w, batcheddata)
     n = 4
     matchedCount = zeros(n)
     totalCount = zeros(n)
     for data in batcheddata
          x, xbatchsizes, y, ybatchsizes, yorder = data
          scores = Array(predict(w, x, xbatchsizes, y, ybatchsizes, yorder))
          ind = findmax(scores,1)[2]
          ind = map(x->ind2sub(scores, x)[1], ind)[:]
          batchsizes = data[4]
          source = reArrange(ind, ybatchsizes)
          target = reArrange(y, ybatchsizes)
          for i = 1:n
               for j = 1:length(source)
                    matchedCount[i] += nGramCounts(source[j], target[j], i)[1]
                    totalCount[i]  += nGramCounts(source[j], target[j], i)[2]
               end
          end
     end
     logBleu = 0
     for i = 1:n
          logBleu += 1/n * log(matchedCount[i]/totalCount[i])
     end
     return exp(logBleu)
end

function tstlossCalc(w, batcheddata)
     totalloss = 0.0
     sentencecount = 0
     for data in batcheddata
          totalloss += loss(w, data)
          sentencecount += data[2][1]
     end
     return totalloss / sentencecount
end



function main()
     srand(1234)
     vocabsize = 30000
     embeddingsize = 620
     hiddensize = 1000
     batchsize = 80
     sentencelength = 50
     maxoutsize = 500
     epocs = 20
     trn, dev, tst, EnVocab, FrVocab = datainit(vocabsize, sentencelength)
     batcheddata = minibatch(trn,batchsize)
     batchedtst = minibatch(tst,batchsize)
     w = modelinit(vocabsize,embeddingsize,hiddensize,maxoutsize)
     shuffle!(batcheddata)
     for i = 1:epocs
          lossinepoch = train(w, batcheddata)
          info("BLEU Score is being calculated.")
          bleuScoretrn = bleuCalc(w, batcheddata) * 100
          bleuScore = bleuCalc(w, batchedtst) * 100
          trnloss = tstlossCalc(w, batcheddata)
          tstloss = tstlossCalc(w, batchedtst)
          println((:epoch, i, :trnloss, lossinepoch, :trnloss ,trnloss, :tstloss, tstloss,:bleuscoretrn, bleuScoretrn ,:bleuscoretst, bleuScore))
     end
end

main()
