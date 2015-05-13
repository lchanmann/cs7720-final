function pca_2_plot()
% PCA_2_PLOT - generate 2d plot for 
%       dataset from PCA method

display(' ');
display('Generating plot (PCA). Press any key to continue...');
pause();

load 'dataset_pca_2.mat';

X = X_pca(:, 2:end);
y = X_pca(:, 1);

[~, test] = data_partition(X, y);

y = test(:, 1);
x1 = test(:, 2);
x2 = test(:, 3);

plot2(x1, x2, y);