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

% This is for Linear Regression 


features = size(theta)(1);
h_x = X*theta;
J = sum((h_x-y).^2)/(2*m) + sum(theta(2:features).^2)*lambda/(2*m);

for j = 1:size(theta)(1)
    if j == 1 
        grad(j) = 1/m * X(:,j)'*(h_x - y);
    else 
        grad(j) = 1/m * X(:,j)'*(h_x - y)+ lambda*theta(j)/m;
    end
end









% =========================================================================

grad = grad(:);

end
