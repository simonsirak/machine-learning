# Introduction
This lecture introduced the topic of ML. 

## Usefulness
ML is useful when it is too difficult to model mathematical rules, when data is available and when a pattern exists.

## Classification
We train some model on labeled training data so that it learns to classify it. The trained model then tries to classify
unseen data.

## Features
Features are the input that we record. A feature could be a pixel, for instance.

## Nearest Neighbor (NN)
For a given unseen input, calculate the distance in faeture space to all training inputs. Then take a majority vote for the
labels associated with the K nearest neighbors. For 1-NN, the training error will always be 0 since a training input is 
always closest to itself.

Lower K means we believe more in the training data, and may result in overfitting and higher variance. Higher K leads to a smoother, more 
generalized boundary between the classes, but may result in a boundary that neglects too much of the impact of individual 
data points, leading to higher bias.

Classification is linear in training data and dimension of an input. But requires storage of all training set.
