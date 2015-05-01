startup

% Train and test data separation
y = dataset(:, 1);
X = dataset(:, 3:end);

% Export full dataset
X_full = [y X];
save('dataset_full', 'X_full');

% Export dataset with PCA dimension reduction
% chosen dimensions: [1, 2, 4, 8, 12]
M = [1 2 4 5 6];
for m=M
    [~,~,~,~,W] = PCA(X', [], m);
    X_pca = [y (W * X')'];
    
    save(['dataset_pca_', num2str(m), '.mat'], 'X_pca');
end

%% Plot 2D projected dataset
clc; clear all; close all;
load 'dataset_pca_2.mat';

y = X_pca(:, 1);
x1 = X_pca(:, 2);
x2 = X_pca(:, 3);

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

%% Neural Nets with 5-D projected data
clc; clear all; close all;
load 'dataset_pca_5.mat'

train_x = X_pca(:, 2:end);
y = X_pca(:, 1);
[r, d] = size(train_x);
C = unique(y)';
train_y = (y * (1 ./ C) == ones(r, length(C)));
H = round( (d + length(C)) * 2/3 ); % the number of nodes in each hidden layers

test_x = train_x;

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

nn = nnsetup([d H length(C)]); % nn structure [input, hidden, ..., hidden, output]
nn.activation_function = 'sigm';
% nn.learningRate = 1; % Should decrease over time.

opts.numepochs = 300;
opts.batchsize = 10; 
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, train_y);
display(er);

%% Deep Belief Network with 5-D projected data
clc; clear all; close all;
load 'dataset_pca_5.mat'

train_x = X_pca(:, 2:end);
y = X_pca(:, 1);
[r, d] = size(train_x);
C = unique(y)';
train_y = (y * (1 ./ C) == ones(r, length(C)));
H = 6; %round( (d + length(C)) * 2/3 );

% normalize the data to [0..1]
% as it is required
x_min = min(train_x);
x_max = max(train_x);
train_x = (train_x - ones(r, 1) * x_min) ./ (ones(r, 1) * (x_max-x_min));

test_x = train_x;

dbn.sizes = [H H]; % hidden nodes of hidden layers
opts.numepochs = 10;
opts.batchsize = 10;
opts.momentum  = 0;
opts.alpha     = 1; % Learning rate
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%%unfold dbn to nn
nn = dbnunfoldtonn(dbn, length(C));
nn.activation_function = 'sigm';

%train nn
opts.numepochs = 10;
opts.batchsize = 10;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, train_y);

display(er);