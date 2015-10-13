function [mu, E, V] = AC_pca (X)

% performs PCA, does not assume that X is centered
% X has size n*p , where n is the number of samples and p the dim
% each row of X must represent a sample
% mu is the mean (size 1*p)
% E is the sequence of eigenvalues of the covariance matrix, in decreasing
% order
% V is the sequence of eigenvectors: each COLUMN of V is an eigenvector !

%parameter
n = size(X,1);

% compute the mean
mu = mean(X,1);
% get the covariance
C = X-repmat(mu,[n 1]);
C = (1/n)*(C')*C;
% diagonalize it
[W,D] = eig(C);
% reshape everything
E = diag(D);
[E,I] = sort(E,'descend');
V = W(:,I);
end
