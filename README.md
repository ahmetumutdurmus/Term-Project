# Term-Project

  This repository contains the code necessary for the replication of much celebrated work "Bahdanau, D., Cho, K., and Bengio, Y. 2014. Neural machine translation by jointly learning to align and translate. In ICLR 2015." The Encoder-Decoder mechanism with alignment as proposed in Bahdanau et al.(2015) is the workhorse model of state of the art machine translators. 
The code is written in Julia and a deep learning package Knet.jl is used as well. Knet.jl basically provides an AutoGrad function which takes care of the tedious backpropagation operations and also provides libraries for GPU support. Here is a link to the documentation of Knet:

http://denizyuret.github.io/Knet.jl/latest/index.html

I personally did not have access to a GPU machine and thus went out and used the EC2 machines rented by the Amazon Web Services (AWS). Here are a very useful set of tutorials for those who do not know what EC2 and AWS are:

https://www.youtube.com/watch?v=LKStwibxbR0&list=PLv2a_5pNAko2Jl4Ks7V428ttvy-Fj4NKU
https://www.youtube.com/watch?v=BDBvHOaaKHo&list=PLv2a_5pNAko0Mijc6mnv04xeOut443Wnk

The first link is to a set of online video lectures called the AWS Concepts. AWS Concepts provide a set of videos that in total take about an hour and provide a very brief, general and useful overview of AWS. The second link is the longer follow-up course of the introductory AWS Concepts course and is called the AWS Essentials. I recommend going through all of the first set of videos but reckon videos 1-7 and 26-32 of the AWS Essentials will suffice for our purposes.  
