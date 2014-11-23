HashSGD
=======

Learn multi-label logistic regression model with feature hashing and adaptive learning rate.

This is my experimental code for [Tradeshift Text Classification contest on Kaggle](http://www.kaggle.com/c/tradeshift-text-classification).
I took the solution code in [this post](http://www.kaggle.com/c/tradeshift-text-classification/forums/t/10537/beat-the-benchmark-with-less-than-400mb-of-memory),
and modularize it with a little more functionality for experiments, including

  * plugable different feature transformations
  * data shuffling with fixed memory usage
