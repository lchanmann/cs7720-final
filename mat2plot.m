function [ output_args ] = mat2plot( M )
%creates a 3D graph for matrix data
[n,m] = size(M);
x=[];
y=[];
z=[];
for(i=1:m)
    z= cat(1,z,M(:,i));
    x= cat(1,x,[1:n]');
    y= cat(1,y,repmat(i,n,1));
end

scatter3(x,y,z,'filled')
end
