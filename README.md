# Cluster Validation with VIC

VIC is a Cluster Validation technique that uses a set of classifiers to evaluate a given partition of a data set.

This implementation uses a custom database of scientometrics data for the QS 2019 top Universities world wide. The data was gathered from Scopus and Scival.

VIC is a method that uses a set of supervised classifiers and k-fold cross validation to evaluate a partition the algorithm works likewise.

* set v = 0
* For each classifier.
    * Set v' = 0
    * Divide data in k folds
    * For each fold.
        * Train with four remaining folds and test AUC with current fold.
        * Update v' = v'+AUC(this_fold)
    * Update v = max(v,v'/k)

## The Classifiers

This implementation contains 5 classifiers:

1. Random Forest
1. Support Vector Machine
1. Naive Bayes
1. Linear Discriminant Analysis
1. Gradient Boosting

## Adding New Classifiers

This repository follows Scikit-Learn's workflow and thus the easiest way to add a model is by using sklearn implementations. You need to add an identifier to the list in line 30 of `vic.py` and then add the corresponding line/lines to select and define the classifier in the function `train_and_test()` defined in `models.py`. If it's not a model from sklearn you must create a Class that includes the methods `fit(x,y)` and `predict(x)`, that train the model and make inference over some input correspondingly. Two examples using this method for custom classifiers are included; a Logistic Regression model in numpy and a Multi Layer Perceptron in TensorFlow (if you want to try MLP un comment the corresponding class an condition in `models.py`).

## Usage

To test the current implementation all you need is:

`python vic.py`

The program will then run VIC on 50 partitions of our QS Dataset with cluster separations between rank 75 and 125. A report is generated showing the value of v for each partition and the classifier with the highest accuracy.

There are two options available, if you want to try some other data use the flag `--clusters_path` to specify a directory with one or more data sets in 'csv' format. You can also use the `--outfile` flag to specify a new name for the generated report, default is `./vic_report.txt`.
