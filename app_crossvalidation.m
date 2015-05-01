% Load dataset
dataset = load('leaf.csv');

% Train and test data separation
y = dataset(:, 1);

%class sizes
a=histcounts(y);

for i=1:min(a(a~=0))    %we shouldn't subdivide a class into more slices than it has samples
    [T, V] = crossValidate(dataset(3:end),dataset(:,2),i);
    %DO SOMETHING
end
