startup

y = dataset(:, 1);
X = dataset(:, 3:end);

% Feature selection
% [patterns, targets, pattern_numbers] = Exhaustive_Feature_Selection(X', y', '[2,''LS'',[]]');
N_feature = [2 3 4 5 6 7 8 9 10 11 12 13];
for n = N_feature
    X_new = [y Sequential_Feature_Selection(X', y', ['[''Forward'',', num2str(n) , ',''LS'',[]]'])'];
    save(['dataset_', num2str(n), '_features',  '.mat'], 'X_new');
end

X_full = [y X];
% Export full dataset
save('dataset_full', 'X_full');

% Export dataset with PCA dimension reduction
[train, ~] = data_partition(X, y);
train_X = train(:, 2:end);

% chosen dimensions: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
M = [1 2 3 4 5 6 7 8 9 10 11 12 13];
for m=M
    [~,~,~,~,W] = PCA(train_X', [], m);
    X_pca = [y (W * X')'];
    
    save(['dataset_pca_', num2str(m), '.mat'], 'X_pca');
end

% Plot 2D selected feature test dataset
feature_selection_2_plot();

% Plot 2D projected test dataset
pca_2_plot();

%% Experiment with Neural Nets with PCA projected data
clc; clear all; close all;

alpha = [4 2 3 3 2 2 1 2 1 1 1 1 1];
Err = zeros(1, 13);
for i = 1:13
    load(['dataset_pca_', num2str(i) ,'.mat']);

    X = X_pca(:, 2:end);
    y = X_pca(:, 1);

    [train, test] = data_partition(X, y);

    train_x = train(:, 2:end);
    train_y = train(:, 1);
    [r, d] = size(train_x);
    C = unique(train_y)';
    train_y = (train_y * (1 ./ C) == ones(r, length(C)));
    H = round(length(train_x) / (length(C) + d) * (length(train_x) / length(X)));

    test_x = test(:, 2:end);
    test_y = test(:, 1);
    [r, ~] = size(test_x);
    test_y = (test_y * (1 ./ C) == ones(r, length(C)));

    % normalize
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);

    rand('state', 0); % fix the initial weight

    nn = nnsetup([d H length(C)]);          %  nn structure [input, hidden, ..., hidden, output]    
    nn.activation_function = 'tanh_opt';    %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate = alpha(i);             %  Learning rate
    nn.scaling_learningRate = 0.999;        %  Scaling factor for the learning rate (each epoch)
    %     nn.momentum = 0.5;

    opts.numepochs = 1000;
    opts.batchsize = 20; % [10, 14, 20]
    [nn, L] = nntrain(nn, train_x, train_y, opts);

    [er, bad] = nntest(nn, test_x, test_y);
    display(['er = ', num2str(er), ' (', num2str(i) ,'d PCA)' sprintf('\t\t[H=%d, alpha=%d]', H, alpha(i))]);
    Err(i) = er;
end

figure;
plot(1:13, Err);
title('Feed-forward Neural nets with PCA');
xlabel('Dimensions');
ylabel('Error');

%% Experiment with Deep Belief Network with 5-D projected data
clc; clear all; close all;
load 'dataset_pca_4.mat'

X = X_pca(:, 2:end);
y = X_pca(:, 1);

[train, test] = data_partition(X, y);

train_x = train(:, 2:end);
train_y = train(:, 1);
[r, d] = size(train_x);
C = unique(train_y)';
train_y = (train_y * (1 ./ C) == ones(r, length(C)));
H = fix(length(train_x) / (length(C) + d)) - 1;

% normalize the data to [0..1]
% as it is required
train_x = softmax(train_x);
% x_min = min(train_x);
% x_max = max(train_x);
% train_x = (train_x - ones(r, 1) * x_min) ./ (ones(r, 1) * (x_max-x_min));

test_x = test(:, 2:end);
test_y = test(:, 1);
[r, ~] = size(test_x);
test_y = (test_y * (1 ./ C) == ones(r, length(C)));

rand('state', 0);
dbn.sizes = [H]; % hidden nodes of hidden layers
opts.numepochs = 200;
opts.batchsize = 20;
opts.momentum  = 0.5;
opts.alpha     = 1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%%unfold dbn to nn
nn = dbnunfoldtonn(dbn, length(C));
nn.activation_function = 'tanh_opt';
% nn.output = 'softmax';
nn.learningRate = 1; % Should decrease over time.
nn.scaling_learningRate = 0.999;

%train nn
opts.numepochs = 100;
opts.batchsize = 20;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

display(er);

%% Experiment with Deep Belief Network with original dataset
clc; clear all; close all;
load 'dataset_full.mat'

X = X_full(:, 2:end);
y = X_full(:, 1);
[train, test] = data_partition(X, y);

train_x = train(:, 2:end);
train_y = train(:, 1);
[r, d] = size(train_x);
C = unique(train_y)';
train_y = (train_y * (1 ./ C) == ones(r, length(C)));
H = 6; %round( (d + length(C)) * 2/3 ); % the number of nodes in each hidden layers

% normalize the data to [0..1]
% as it is required
x_min = min(train_x);
x_max = max(train_x);
train_x = (train_x - ones(r, 1) * x_min) ./ (ones(r, 1) * (x_max-x_min));

test_x = test(:, 2:end);
test_y = test(:, 1);
[r, ~] = size(test_x);
test_y = (test_y * (1 ./ C) == ones(r, length(C)));

dbn.sizes = [H H]; % hidden nodes of hidden layers
opts.numepochs = 8;
opts.batchsize = 1;
opts.momentum  = 0;
opts.alpha     = 1; % Learning rate
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%%unfold dbn to nn
nn = dbnunfoldtonn(dbn, length(C));
nn.activation_function = 'sigm';
nn.learningRate = 1; % Should decrease over time.
nn.scaling_learningRate = 0.999;

%train nn
% opts.numepochs = 18;
% opts.batchsize = 1;
[nn, L] = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

display(er);

%% Experiment with Neural Networks with feature selection dataset
clc; clear all; close all;

alpha = [0 4 6 1 2 1 2 2 1 1 1 6 1];
Err = zeros(1, 13);
for i = 2:13
    load(['dataset_', num2str(i) ,'_features.mat']);
    
    X = X_new(:, 2:end);
    y = X_new(:, 1);
    [train, test] = data_partition(X, y);

    train_x = train(:, 2:end);
    train_y = train(:, 1);
    [r, d] = size(train_x);
    C = unique(train_y)';
    train_y = (train_y * (1 ./ C) == ones(r, length(C)));
    H = round(length(train_x) / (length(C) + d) * (length(train_x) / length(X)));

    test_x = test(:, 2:end);
    test_y = test(:, 1);
    [r, ~] = size(test_x);
    test_y = (test_y * (1 ./ C) == ones(r, length(C)));

    % normalize
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);

    rand('state', 0); % fix the initial weight
    
    nn = nnsetup([d H length(C)]); % nn structure [input, hidden, ..., hidden, output]
    nn.activation_function = 'tanh_opt';
    nn.learningRate = alpha(i); % Should decrease over time.
    nn.scaling_learningRate = 0.999;

    opts.numepochs = 1000;
    opts.batchsize = 20;
    [nn, L] = nntrain(nn, train_x, train_y, opts);

    [er, bad] = nntest(nn, test_x, test_y);
    display(['er = ', num2str(er), ' (', num2str(i) ,' features)' sprintf('\t\t[H=%d, alpha=%d]', H, alpha(i))]);
    Err(i) = er;
end

figure;
plot(2:13, Err(2:13));
title('Feed-forward Neural nets with Feature selection');
xlabel('Dimensions');
ylabel('Error');
ylim();

%% Experiment with Bayesian parameter estimation with
%  5 features dataset
clc; clear all; close all;
load 'dataset_5_features.mat'

X = X_new(:, 2:end);
y = X_new(:, 1);
[train, ~] = data_partition(X, y);

train_x = train(:, 2:end);
train_y = train(:, 1);

[~, d] = size(X);
K = unique(y);
Sigma = zeros(length(K), d, d); % Covariance for each class
for i = 1:length(K)
    X_given_y = train_x(train_y == K(i),:);
    [~, Sigma(i,:,:)] = mle(X_given_y);
end

[mu, Sigma] = Bayesian_parameter_est(train_x', train_y', Sigma);

display(mu);
display(Sigma);