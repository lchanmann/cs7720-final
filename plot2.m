function plot2( x1, x2, y )
% PLOT2 - draw 2D plot from leaf data

figure;
K = unique(y);
markers = '.ox+*sdv^<>pd';
L = {}; % legend

for k = K'
    X1 = x1(y == k);
    X2 = x2(y == k);
    index = fix(1 + (length(markers)-1) * rand);
    marker = markers(index);
    
    scatter(X1, X2, marker); hold on
    L{end+1} = ['C', num2str(k)];
end
hold off
title('2D test dataset (Feature selection)');
legend(L);
