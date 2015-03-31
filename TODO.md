TODO
===

Choosing the dataset to work on:
---
The adult dataset has a mixture of discrete and continuous variance which could not be fit into multivariate gaussian and exceed the topics have been discussed so far.
* Task: find new dataset which do not have discrete random variable.

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
