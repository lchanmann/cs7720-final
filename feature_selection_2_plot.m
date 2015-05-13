function feature_selection_2_plot()
% FEATURE_SELECCTION_2_PLOT - generate 2d plot for 
%       dataset from feature selection method

display(' ');
display('Generating plot (feature selection). Press any key to continue...');
pause();

clc; clear all; close all;
load 'dataset_2_features.mat';

X = X_new(:, 2:end);
y = X_new(:, 1);

[~, test] = data_partition(X, y);

y = test(:, 1);
x1 = test(:, 2);
x2 = test(:, 3);

plot2(x1, x2, y);
