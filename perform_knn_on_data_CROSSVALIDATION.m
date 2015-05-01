% Load dataset
dataset = load('leaf.csv');

class    = dataset(:, 1);   %true class of each sample
specimen = dataset(:, 1);   %sample ID within class

% PCA dimension reduction
M = [1 2 4 8];
for m=M
    load(['dataset_pca_', num2str(m), '.mat'], 'X_pca');    
    [train, test] = crossValidate(X_pca,specimen,1,2);
    [train_class, true_test_class] = crossValidate(class,specimen,1,2);
    learned_test_class = trainKNN(train, train_class,test,3,true_test_class);
end