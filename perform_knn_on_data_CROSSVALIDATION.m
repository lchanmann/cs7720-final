% PCA dimension reduction
M = [1 2 4 5 6];
accuracy = zeros(max(M), 8);
test_set_size=60;
learned_test_class = zeros(max(M),test_set_size)
for m=M
    load(['dataset_pca_train_', num2str(m), '.mat'], 'X_pca_train');    
    load(['dataset_pca_', num2str(m), '.mat'], 'X_pca');    
    for(k=1:8)
        crossval_class = zeros(4, size(X_pca_train,1));
        for(slice=1:4)
            [train, ~] = crossValidate(X_pca_train, specimen, slice, 2);
            crossval_class(slice,:) = Nearest_Neighbor(train(:, 2:end)', train(:, 1)', X_pca_train(:, 2:end)', k);
        end
        learned_train_class = mode(crossval_class(slice,:));
        learned_test_class(m) = Nearest_Neighbor(learned_train_class(:, 2:end)', learned_train_class(:, 1)', X_pca(:, 2:end)', k);;
        true_test_class = X_pca(:, 1);
        accuracy(m,k) = sum(learned_test_class' == true_test_class)/length(true_test_class);
    end
end