% Multivariate Linear Regression 
% Author : Mohanen

% Init 
clear ; close all; clc

% Step 0 : Load the data
% ------------------------
data = load('sample.txt');
% Add intercept term to X
X = data(:, 1:2);
y = data(:, 3);


% Step 1 : Find theta using Normal Equation
% -------------------------------
ne_theta = normal_equation(X, y);
% Display gradient descent's result
fprintf('\nTheta from normal_equation:\t');
fprintf('%f\t', ne_theta);


% Step 2 : Find theta using Gradient Descent
% -------------------------------
[mu sigma gd_theta] = gradient_descent(X, y);
% Display gradient descent's result
fprintf('\nTheta from gradient_descent:\t');
fprintf('%f\t', gd_theta);


% Step 3 : Test and compare the results
% -----------------------------------------------
test_data = load('test.txt');
Xt = [ones(size(test_data(:, 1:2),1), 1) test_data(:, 1:2)];

fprintf('\n\nRunning test data:\n');

for iter = 1:size(Xt,1)
	% normal equation prediction
	ne_predicted = Xt(iter, :) * ne_theta;

	% gradient descent prediction
    for feature_itr = 2:size(Xt,2) 				% Get the size of features and iterate
    	Xt(:,feature_itr)= (Xt(:,feature_itr)-mu(:,feature_itr-1))/sigma(:,feature_itr-1);
    end
    gd_predicted = Xt(iter, :) * gd_theta;

    fprintf('\ninput features\t\t= ');
    fprintf('%f\t', X(iter,:));
    fprintf('\nnormal_equation\t\t= %f \ngradient_descent\t= %f \n', ne_predicted, gd_predicted);
end