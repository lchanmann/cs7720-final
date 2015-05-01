function test_targets = Nearest_Neighbor(train_patterns, train_targets, test_patterns, Knn)

% Classify using the Nearest neighbor algorithm
% Inputs:
% 	train_patterns	- Train patterns (data values of samples)
%	train_targets	- Train targets  (known classes of train samples)
%   test_patterns   - Test  patterns (validation samples)
%	Knn		        - Number of nearest neighbors 
%
% Outputs
%	test_targets	- Predicted targets

L			= length(train_targets);
Uc          = unique(train_targets);

if (L < Knn),
   error('You specified more neighbors than there are points.')
end

N               = size(test_patterns, 2);
test_targets    = zeros(1,N); 
for i = 1:N,
    %%%%%%revised
    if size(test_patterns,1)~=1
        dist            = sum((train_patterns - test_patterns(:,i)*ones(1,L)).^2);
    else
        dist  =         ((train_patterns - test_patterns(:,i)*ones(1,L)).^2);
    end
    %%%%%revised end
    [m, indices]    = sort(dist);
    
    n               = hist(train_targets(indices(1:Knn)), Uc);
    
    [m, best]       = max(n);
    
    test_targets(i) = Uc(best);
end
