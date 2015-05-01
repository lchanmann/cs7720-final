function [training, validation] = crossValidate( data, specimen, slice, slice_size)
%Gets the desired SLICE of a cross-validation round, by leaving out of the 
%TRAINING set all members in SPECIMEN matching that SLICE number, and 
%putting them in the VALIDATION set. That is, it removes one element from
%each class to build the validation set.
    index = false(size(specimen));
    for(i = (slice-1)*slice_size+1 : slice+slice_size)
        index = index | specimen==i;
    end
    training = data(~index,:);
    validation = data(index,:);
end

