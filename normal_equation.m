% Multivariate Linear Regression - normal equation
% Author : Mohanen

function [theta] = normal_equation(X, y)
    m = length(y); 			% number of training examples
    X = [ones(m, 1) X];
	theta = pinv(X' * X) * X' * y;
end