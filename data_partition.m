function [ train, test ] = data_partition( X, target )
% data_partition - Split dataset into train and test set

    C = unique(target)';
    
    train = [];
    test = [];
    for c = C
        Data_given_c = [target(target == c) X(target == c, :)];
        train = cat(1, train, Data_given_c(1:end-2, :));
        test = cat(1, test, Data_given_c(end-1:end, :));
    end
end

