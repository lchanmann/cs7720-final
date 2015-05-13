function save_bayesian_params( train_x, train_y, output )
% SAVE_BAYESIAN_PARAMS - Save bayesian parameters to ".mat" file

[~, d] = size(train_x);
K = unique(train_y);
Sigma = zeros(length(K), d, d); % Covariance for each class
for i = 1:length(K)
    X_given_y = train_x(train_y == K(i),:);
    [~, Sigma(i,:,:)] = mle(X_given_y);
end

[mu, Sigma] = Bayesian_parameter_est(train_x', train_y', Sigma);

save(['mu_', output, '.mat'], 'mu');
save(['Sigma_', output, '.mat'], 'Sigma');

