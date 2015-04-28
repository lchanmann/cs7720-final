function [ m, P ] = mle( dataset )
% mle - Maximum likelihood estimator for mean and covariance 
%       of 1-D and 2-D Gaussian dataset
%
%   m : the estimated mean (sample mean)
%   P : the estimated biased variance for 1-D dataset
%           and covariance matrix for 2-D dataset
%
% Note:
%   P = [var1 cov(1,2); cov(1,2) var2]
% 
% where
%   var1        - biased variance of x1
%   cov(1, 2)   - E[(x1-mean_x1)(x2-mean_x2)]
%   var2        - biased variance of x2
%
    m = mean(dataset)';
    P = cov(dataset, 1);
end

