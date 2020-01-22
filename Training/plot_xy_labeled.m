function plot_xy_labeled(X,Y)

class0 = X(Y == 0,:);
class1 = X(Y == 1,:);

figure;
plot(class1(:,1),class1(:,2), '.');
hold on
plot(class0(:,1),class0(:,2), '.');
hold off


end

