# Random-Forest
We aim to build a spam classifier using a CART model

First, we build a CART model. To build a CART model, we use UCR email spam dataset which can be found : https://archive.ics.uci.edu/ml/datasets/Spambase. Note that in this step we did not prune the classification tree.

Second, we build a Random Forest model. We randomly shuffle the data and use 75% of the data as the training sample and 25% of the data as the testing sample. Next, we compare test error for our classification tree and random forest models on testing data. 

Next, we fit a series of random forest classifiers to the data to explore sensitivity to the parameter nu (the number of variables selected at random to split).

Finally, we use one-class SVM for spam filtering using RBF kernel. 

