function yguess = pca_nn(Xtrain,ytrain,Xtest,k)
%Following instructions of  pdf to get the X_reduced for train
Vtrain = pca(Xtrain);
Vtrain_k = Vtrain(:, 1:k);

ntrain = size(Xtrain, 1);

one_ntrain = ones(ntrain, 1);

xtrain_bar = (1 / ntrain) * Xtrain' * one_ntrain;
Xtrain_c = Xtrain - (one_ntrain * xtrain_bar');
Xtrain_reduced = Xtrain_c * Vtrain_k;

%Following instructions of  pdf to get the X_reduced for test
Vtest = pca(Xtest);
Vtest_k = Vtest(:, 1:k);

ntest = size(Xtest, 1);

one_ntest = ones(ntest, 1);

xtest_bar = (1 / ntest) * Xtest' * one_ntest;
Xtest_c = Xtest - (one_ntest * xtest_bar');
Xtest_reduced = Xtest_c * Vtest_k;

%yguess = nearest_neighbor(Xtrain_reduced, ytrain, Xtest_reduced);
yguess = zeros(size(Xtest_reduced, 1),  1);
for i = size(Xtest_reduced, 1)
    dist = vecnorm(repmat(Xtest_reduced(i, :), size(Xtrain_reduced, 1), 1) - Xtrain_reduced, 2, 2);
    [dmin, argmin] = min(dist);
    yguess(i) = ytrain(argmin);
end
end