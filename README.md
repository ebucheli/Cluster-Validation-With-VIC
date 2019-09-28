# Cluster Validation with VIC

VIC is a Cluster Validation technique that uses a set of classifiers to evaluate a given partition of a data set.

This implementation uses a custom database of scientometrics data for the QS 2019 top Universities world wide. The data was gathered from Scopus and Scival.

VIC is a method that uses a set of supervised classifiers and k-fold cross validation to evaluate a partition the algorithm works likewise.

1. set $v = 0$
1. For each classifier.
1. Set $v' = 0$
1. Divide data set in $k$ folds and train $k$ models.
1. After training each model update $v' = v'+AUC()
1. Update v = max(v,v'/k)

## The Classifiers

This implementation contains 5 classifiers:

1. Random Forest
1. Support Vector Machine
1. Naive Bayes
1. Linear Discriminant Analysis
1. Gradient Boosting

However it is easy to add new classifiers. This repository follows Scikit-Learn's workflow and thus the easiest way to add a model is by using sklearn implementations. You need to add an identifier to the list in line 30 of `vic.py` and then add the corresponding line/lines to define the classifier in the function `train_and_test()` defined in `models.py`. If it's not a model from sklearn you must create a Class with the methods fit and predict, that train the model and make inference over some input correspondingly. An example MLP programed in TensorFlow is included as an example.

## Usage
