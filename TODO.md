TODO
===

Choosing the dataset to work on:
---
The adult dataset has a mixture of discrete and continuous variance which could not be fit into multivariate gaussian and exceed the topics have been discussed so far.
* Task: find new dataset which do not have discrete random variable.

Datesets to be considered:
---
1. [Engery efficiency](http://archive.ics.uci.edu/ml/datasets/Energy+efficiency "Two continuous output variables")
2. [Distinguish hand written 4 and 9](http://archive.ics.uci.edu/ml/datasets/Gisette "Big dataset and has been studied extensively")
3. [Ionosphere](http://archive.ics.uci.edu/ml/datasets/Ionosphere "Good dataset")
4. [Isolet](http://archive.ics.uci.edu/ml/datasets/ISOLET "Spoken letter recognition")
5. [Leaf](http://archive.ics.uci.edu/ml/datasets/Leaf "40 leaf classification")
6. [LSVT Voice Rehabilitation](http://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation "Just another")
7. [Lung Cancer](http://archive.ics.uci.edu/ml/datasets/Lung+Cancer "Just another")

Questions to be clarified
---

  1. using Adult dataset to predict whether the income exceeds $50k/ys: change dataset
  2. data pre-processing (training and test data)
    * Manually using Excel: is ok in experimentation phase
    * Programming using Matlab: recommended
  3. dealing with missing data: remove
  4. gaussian assumption
    * MLE: is fine for the project
    * MAP (guessing mu_0 and Sigma_0): need hyper-parameter estimation
  5. related works (papers): change dataset
