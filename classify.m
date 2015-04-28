function [ c ] = classify( X, Theta_1, Theta_2, Theta_3 )
% classify - Classify X given Theta {1 2 and 3}
%   by comparing the value of discriminant function g(x)
%   for each parameter theta.
%
% Return:
%   c - classification vector
%
% where value of
%   c = 1 (Iris-setosa)
%   c = 2 (Iris-versicolor)
%   c = 3 (Iris-virginica)
%

    [r, ~] = size(X);
    c = zeros(r, 1);
    
    mu_column = 1;
    sigma_column = 2:3;
    
    for k=1:r
        x = X(k, :)';
        [~, c(k,:)] = max([
            g_mle(x, Theta_1(:, mu_column), Theta_1(:, sigma_column)) ...
            g_mle(x, Theta_2(:, mu_column), Theta_2(:, sigma_column)) ...
            g_mle(x, Theta_3(:, mu_column), Theta_3(:, sigma_column)) ]);
    end
end

