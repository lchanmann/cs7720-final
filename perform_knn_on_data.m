% PCA dimension reduction
features = 13;
neighbors = 12;

accuracy_PCA = zeros(features, neighbors);
accuracy_features = zeros(features, neighbors);

dataset = load('leaf.csv');
y = dataset(:, 1);

for m=2:features
    %user feedback
    disp(['Performing KNN with ', num2str(m), ' features']);
    
    load(['dataset_pca_train_', num2str(m), '.mat'], 'X_pca_train');    
    load(['dataset_pca_', num2str(m), '.mat'], 'X_pca');    
    load(['dataset_', num2str(m), '_features.mat'], 'X_new');    
    [X_new_train, X_new_test] = data_partition(X_new(:, 2:end), y);
    for(k=1:neighbors)
        learned_test_PCA_class = Nearest_Neighbor(X_pca_train(:, 2:end)', X_pca_train(:, 1)', X_pca(:, 2:end)', k);
        learned_test_features_class = Nearest_Neighbor(X_new_train(:, 2:end)', X_new_train(:, 1)', X_new_test(:, 2:end)', k);
        true_test_class_PCA = X_pca(:, 1);
        true_test_class_features = X_new_test(:, 1);
        accuracy_PCA(m,k) = sum(learned_test_PCA_class' == true_test_class_PCA)/length(true_test_class_PCA);
        accuracy_features(m,k) = sum(learned_test_features_class' == true_test_class_features)/length(true_test_class_features);
    end
end
%eliminate temp variables
clear X_pca_train X_pca X_new X_new_train X_new_test
clear learned_test_PCA_class learned_test_features_class true_test_class_PCA true_test_class_features

disp('Thank you for choosing KNN. Accuracy results may be seen in accuracy_PCA and accuracy_features')