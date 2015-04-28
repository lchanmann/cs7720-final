function [ g ] = g_mle( x, mu, Sigma )
% g_mle - Discriminant function 'g' for Maximum Likelihood Estimation
%   Compute log( P(x|w) )
%
% Input:
%   mu - mean of P(x|w)
%   Sigma - covariance matrix of P(x|w)

    d = length(x);
    x_tilde = x - mu;
    g = - d/2*log(2*pi) ...
        - 1/2*log(det(Sigma)) ...
        - 1/2*x_tilde'/Sigma*x_tilde;
end

