function learned_test_targets = trainKNN( train_patterns, train_targets, test_patterns, Knn, test_targets )
%Iterates the NEAREST_NEIGHBOR algorithm with the given parameters until a
%fixed point is reached
%Inputs:
% 	train_patterns	- data values of samples
%	train_targets	- known classes of train samples
%   test_patterns   - validation samples
%	Knn		        - Number of nearest neighbors 
%   test_targets    - known classes of validation samples (optional. Used
%                     solely to print error of each iteration)
%
% Outputs
%	learned_test_targets	- Predicted targets
%
    assert( size(train_patterns,2) == size(train_targets,2) );
    assert( nargin < 5 || size(test_patterns,2) == size(test_targets,2) );
    old_test_targets     = zeros(1,size(test_patterns,2));
    learned_test_targets = ones(1,size(test_patterns,2));
    while(~all(learned_test_targets == old_test_targets))
        old_test_targets = learned_test_targets;
        learned_test_targets = Nearest_Neighbor( train_patterns, train_targets, test_patterns, Knn );
        %user feedback
        if nargin >=5
            error = mean(abs(test_targets - learned_test_targets),2)
        end
    end
end

