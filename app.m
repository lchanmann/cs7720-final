clc; clear all; close all;
% Load dataset
dataset = load('leaf.csv');

% Train and test data separation
y = dataset(:, 1);
X = dataset(:, 3:end);

% PCA dimension reduction
M = [1 2 4 8];
for m=M
    [~,~,~,~,W] = PCA(X', [], m);
    X_pca = [y (W * X')'];
    
    save(['dataset_pca_', num2str(m), '.mat'], 'X_pca');
end