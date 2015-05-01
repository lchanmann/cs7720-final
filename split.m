function [training, testing] = split( dataset, class_index, favor_training )
%Divides the dataset into two equal parts, ensuring each part has the same
%number of members of each class
%
%   inputs:
%       dataset        - the dataset to be split
%       class_index    - linear vector indicating which class (numbered 1
%                        through n) each row in DATASET corresponds to
%       favor_training - defines whether the first class with an odd number
%                        of samples should add that sample to the training
%                        set. All odd classes from then on will alternate.
    assert(mod(size(dataset,1),2)==0)
    
    classSize=histcounts(class_index);
    
    if(nargin<3)
        favor_training = false;
    
    training = [];
    testing  = [];
    
    for(i=1:length(classSize))
        class = dataset(class_index==i, :);
        n = size(class,1);
        if(mod(n,2)==0)
            %class is even
            training = cat(1, training, class( 1     : n/2, : ));
            testing  = cat(1, testing,  class( n/2+1 : n  , : ));
        else
            %class is odd. Keep balance
            n=n-1;
            if(favor_training)
                training = cat(1, training, class( 1     : n/2+1, : ));
                testing  = cat(1, testing,  class( n/2+2 : n+1  , : ));
            else
                training = cat(1, training, class( 1     : n/2  , : ));
                testing  = cat(1, testing,  class( n/2+1 : n+1  , : ));
            end
            favor_training = ~favor_training;
        end
    end
end

