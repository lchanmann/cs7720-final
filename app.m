startup

% Train and test data separation
y = dataset(:, 1);
X = dataset(:, 3:end);

% Export full dataset
X_full = [y X];
save('dataset_full', 'X_full');

% Export dataset with PCA dimension reduction
% chosen dimensions: [1, 2, 4, 8, 12]
M = [1 2 4 8 12];
for m=M
    [~,~,~,~,W] = PCA(X', [], m);
    X_pca = [y (W * X')'];
    
    save(['dataset_pca_', num2str(m), '.mat'], 'X_pca');
end

% Plot 2D projected dataset
load 'dataset_pca_2.mat'
y = X_pca(:, 1);
x1 = X_pca(:, 2);
x2 = X_pca(:, 3);

figure;
K = unique(y);
markers = '.ox+*sdv^<>pd';

for k = K'
    X1 = x1(y == k);
    X2 = x2(y == k);
    index = int8(1 + (length(markers)-1) * rand);
    marker = markers(index);
    
    scatter(X1, X2, marker); hold on
end
hold off

%%
test_example_NN