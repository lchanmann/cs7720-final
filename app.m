startup

% Train and test data separation
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
% chosen dimensions: [1, 2, 4, 8, 12]

[train, ~] = data_partition(X, y);
train_X = train(:, 2:end);

M = [1 2 3 4 5 6 7 8 9 10 11 12 13];
for m=M
    [~,~,~,~,W] = PCA(train_X', [], m);
    X_pca = [y (W * X')'];
    
    save(['dataset_pca_', num2str(m), '.mat'], 'X_pca');
end

% MultipleDiscriminantAnalysis from classification toolbox
% will yield error since the number of classes is larger than
% the number of input dimension
%
%[~,~,W] = MultipleDiscriminantAnalysis(X', y');

%% Plot 2D selected feature test dataset
clc; clear all; close all;
load 'dataset_2_features.mat';

X = X_new(:, 2:end);
y = X_new(:, 1);

[~, test] = data_partition(X, y);

y = test(:, 1);
x1 = test(:, 2);
x2 = test(:, 3);

figure;
K = unique(y);
markers = '.ox+*sdv^<>pd';

for k = K'
    X1 = x1(y == k);
    X2 = x2(y == k);
    index = fix(1 + (length(markers)-1) * rand);
    marker = markers(index);
    
    scatter(X1, X2, marker); hold on
end
hold off

%% Plot 2D projected test dataset
clc; clear all; close all;
load 'dataset_pca_2.mat';

X = X_pca(:, 2:end);
y = X_pca(:, 1);

[~, test] = data_partition(X, y);

y = test(:, 1);
x1 = test(:, 2);
x2 = test(:, 3);

figure;
K = unique(y);
markers = '.ox+*sdv^<>pd';

for k = K'
    X1 = x1(y == k);
    X2 = x2(y == k);
    index = fix(1 + (length(markers)-1) * rand);
    marker = markers(index);
    
    scatter(X1, X2, marker); hold on
end
hold off

%% Experiment with Neural Nets with 5-D projected data
clc; clear all; close all;
% load 'dataset_pca_train_5.mat'
load 'dataset_pca_all_5.mat'

X = X_pca_all(:, 2:end);
y = X_pca_all(:, 1);

[train, test] = data_partition(X, y);

train_x = train(:, 2:end);
train_y = train(:, 1);
[r, d] = size(train_x);
C = unique(train_y)';
train_y = (train_y * (1 ./ C) == ones(r, length(C)));
H = 6;%round( (d + length(C)) * 2/3 ); % the number of nodes in each hidden layers

test_x = test(:, 2:end);
test_y = test(:, 1);
[r, ~] = size(test_x);
test_y = (test_y * (1 ./ C) == ones(r, length(C)));

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

nn = nnsetup([d H length(C)]); % nn structure [input, hidden, ..., hidden, output]
nn.activation_function = 'sigm';
nn.learningRate = 1; % Should decrease over time.
nn.scaling_learningRate = 0.999;

opts.numepochs = 110;
opts.batchsize = 1; 
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
display(er);

%% Experiment with Deep Belief Network with 5-D projected data
clc; clear all; close all;
load 'dataset_pca_train_5.mat'
load 'dataset_pca_5.mat'

train_x = X_pca_train(:, 2:end);
train_y = X_pca_train(:, 1);
[r, d] = size(train_x);
C = unique(train_y)';
train_y = (train_y * (1 ./ C) == ones(r, length(C)));
H = 6; %round( (d + length(C)) * 2/3 ); % the number of nodes in each hidden layers

% normalize the data to [0..1]
% as it is required
x_min = min(train_x);
x_max = max(train_x);
train_x = (train_x - ones(r, 1) * x_min) ./ (ones(r, 1) * (x_max-x_min));

test_x = X_pca(:, 2:end);
test_y = X_pca(:, 1);
[r, ~] = size(test_x);
test_y = (test_y * (1 ./ C) == ones(r, length(C)));

dbn.sizes = [H H]; % hidden nodes of hidden layers
opts.numepochs = 20;
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
opts.numepochs = 12;
opts.batchsize = 1;
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

%% Experiment with Neural Networks with original dataset
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

test_x = test(:, 2:end);
test_y = test(:, 1);
[r, ~] = size(test_x);
test_y = (test_y * (1 ./ C) == ones(r, length(C)));

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

nn = nnsetup([d H length(C)]); % nn structure [input, hidden, ..., hidden, output]
nn.activation_function = 'sigm';
nn.learningRate = 1; % Should decrease over time.
nn.scaling_learningRate = 0.999;

opts.numepochs = 110;
opts.batchsize = 1; % length(train_x)/59;
% opts.momentum  = 0.2;
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
display(er);

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