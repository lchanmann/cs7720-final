% Load dataset
dataset = load('leaf.csv');

% Train and test data separation
y = dataset(:, 1);

% PCA dimension reduction
M = [1 2 4 8];
for m=M
    [~,~,~,~,W] = PCA(dataset', [], m);
    X_pca = [y (W * dataset')'];
    
    save(['dataset_pca_', num2str(m), '.mat'], 'X_pca');
end