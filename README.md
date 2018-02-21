CSE474 Introduction to Machine Learning
Programming Assignment 1
Neural Networks
Due Date: March 7th 2018
1 Introduction
In this assignment, your task is to implement a Multilayer Perceptron Neural Network and evaluate its
performance in classifying handwritten digits You will also use the same network to analyze a more challenging
hand-drawn images dataset and compare the performance of the neural network against a deep neural
network using the TensorFlow library.
After completing this assignment, you are able to understand:
  • How Neural Network works and use Feed Forward, Back Propagation to implement Neural Network?
  • How to setup a Machine Learning experiment on real data?
  • How regularization plays a role in the bias-variance tradeoff?
  • How to use TensorFlow library to deploy deep neural networks and understand how having multiple hidden layers can improve the performance of the neural network?
To get started with the exercise, you will need to download the supporting files and unzip its contents to
the directory you want to complete this assignment.
Warning: In this project, you will have to handle many computing intensive tasks such as training
a neural network. Our suggestion is to use the CSE server Metallica (this server is dedicated to intensive
computing tasks) and CSE server Springsteen (this boss server is dedicated to running TensorFlow) to
run your computation. YOU MUST USE PYTHON 3 FOR IMPLEMENTATION. In addition, training
such a big dataset will take a very long time, maybe many hours or even days to complete. Therefore,
we suggest that you should start doing this project as soon as possible so that the computer will have
time to do heavy computational jobs.
