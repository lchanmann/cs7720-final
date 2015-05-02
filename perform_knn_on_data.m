% PCA dimension reduction
M = [2:13];
accuracy_PCA = zeros(max(M), 8);
accuracy_features = zeros(max(M), 8);
for m=M
    load(['dataset_pca_train_', num2str(m), '.mat'], 'X_pca_train');    
    load(['dataset_pca_', num2str(m), '.mat'], 'X_pca');    
    load(['dataset_', num2str(m), 'features.mat'], 'X_new');    
    [X_new_train, ~] = data_partition(X_new);
    for(k=1:12)
        learned_test_PCA_class = Nearest_Neighbor(X_pca_train(:, 2:end)', X_pca_train(:, 1)', X_pca(:, 2:end)', k);
        learned_test_features_class = Nearest_Neighbor(X_new_train(:, 2:end)', X_new_train(:, 1)', X_pca(:, 2:end)', k);
        true_test_class = X_pca(:, 1);
        accuracy_PCA(m,k) = sum(learned_test_PCA_class' == true_test_class)/length(true_test_class);
        accuracy_features(m,k) = sum(learned_test_features_class' == true_test_class)/length(true_test_class);
    end
end