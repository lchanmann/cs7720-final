% Load dataset
dataset = load('leaf.csv');

class    = dataset(:, 1);   %true class of each sample
specimen = dataset(:, 1);   %sample ID within class

% PCA dimension reduction
M = [1 2 4 8];
for m=M
    load(['dataset_pca_', num2str(m), '.mat'], 'X_pca');    
    [train, test] = split(X_pca,specimen);  %if using another implementation of KNN, try switching to crossValidate
    [train_class, true_test_class] = split(class,specimen);
    learned_test_class = trainKNN(train, train_class,test,3,true_test_class);
end