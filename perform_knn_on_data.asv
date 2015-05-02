% Load dataset
dataset = load('leaf.csv');

class    = dataset(:, 1);   %true class of each sample
specimen = dataset(:, 1);   %sample ID within class

% PCA dimension reduction
M = [1 2 4 5 6];
for m=M
    disp(['PCA', num2str(m)]);
    load(['dataset_pca_train_', num2str(m), '.mat'], 'X_pca_train');    
    load(['dataset_pca_', num2str(m), '.mat'], 'X_pca');    
    learned_test_class = trainKNN(X_pca_train(:, 2:end)', X_pca_train(:, 1)', X_pca(:, 2:end)', 3, X_pca(:, 1)');
end