% Load dataset
dataset = load('leaf.csv');

% Train and test data separation
y = dataset(:, 1);

% PCA dimension reduction
M = [8];
for m=M
    [~,~,~,~,W] = PCA(dataset', [], m);
    X_pca = (W * dataset')';
    
    save('dataset_pca.mat', 'X_pca');
end