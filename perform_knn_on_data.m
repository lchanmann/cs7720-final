features = 14;
neighbors = 12;

accuracy_PCA = zeros(features-1, neighbors);
accuracy_features = zeros(features, neighbors);

dataset = load('leaf.csv');
y = dataset(:, 1);

%% Train and compute accuracies
for m=1:features
    %user feedback
    disp(['Performing KNN with ', num2str(m), ' features']);
    
    if(m<14)
        if(m>1)
            load(['dataset_', num2str(m), '_features.mat'], 'X_new');    
            [X_new_train, X_new_test] = data_partition(X_new(:, 2:end), y);
        end
        load(['dataset_pca_', num2str(m), '.mat'], 'X_pca');    
        [X_pca_train,X_pca_test] = data_partition(X_pca(:, 2:end), y);
    else
        [X_new_train, X_new_test] = data_partition(dataset(:, 3:end), y);
    end
    for(k=1:neighbors)
        if(m>1)
            learned_test_features_class = Nearest_Neighbor(X_new_train(:, 2:end)', ...
                X_new_train(:, 1)', X_new_test(:, 2:end)', k);
            true_test_class_features = X_new_test(:, 1);
            accuracy_features(m,k) = sum(learned_test_features_class' == true_test_class_features) ...
                / length(true_test_class_features);
        end
        if(m<14)
            learned_test_PCA_class = Nearest_Neighbor(X_pca_train(:, 2:end)', X_pca_train(:, 1)', ...
                X_pca_test(:, 2:end)', k);
            true_test_class_PCA = X_pca_test(:, 1);
            accuracy_PCA(m,k) = sum(learned_test_PCA_class' == true_test_class_PCA) ...
                / length(true_test_class_PCA);
        end
    end
end

%% Graphs
disp('generating graphs')
figure
surf(1-accuracy_features)
set(gca,'YDir','Reverse')
ylabel('features')
ylim([1 14])
xlabel('neighbors')
xlim([1 12])
title('Error using KNN with Forward Feature Selection')
zlabel('error')
zlim([0 1])
set(gcf, 'InvertHardCopy', 'off');
figure
surf(1-accuracy_PCA)
set(gca,'YDir','Reverse')
ylabel('features')
ylim([1 14])
xlabel('neighbors')
xlim([1 12])
title('Error using KNN with Principal Component Analysis')
zlabel('error')
zlim([0 1])
set(gcf, 'InvertHardCopy', 'off');

%% eliminate temp variables
clear X_pca_train X_pca X_new X_new_train X_new_test
clear learned_test_PCA_class learned_test_features_class true_test_class_PCA true_test_class_features

disp('Accuracy results may be seen in accuracy_PCA and accuracy_features')