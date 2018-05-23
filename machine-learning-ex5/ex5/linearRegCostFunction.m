function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hoft = X*theta;
sumSqErr = sum((hoft - y).^2);

%Don't forget to exclude theta0, as we don't include it in regularisation
jregterm = (lambda/(2*m))*sum(theta(2:end).^2);
gradregterm = (lambda/m).*theta(2:end);

J = sumSqErr/(2*m) + jregterm;
grad = (1/m * (X'*(hoft - y)));
grad(2:end) = grad(2:end) + gradregterm;


% =========================================================================

grad = grad(:);

end
