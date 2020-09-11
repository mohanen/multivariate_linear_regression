% Multivariate Linear Regression - gradient descent
% Author : Mohanen

function [mu sigma theta] = gradient_descent(X, y)

    % Step 1 : Normalize - Feature Scaling
    % -------------------------------------
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));
    m = length(y);                      % number of training examples

    for iter = 1:size(X,2) 				% Get the size of features and iterate
    	mu(:,iter)= mean(X(:,iter)); 	% find the mean of feature for all samples  
    	sigma(:,iter)= std(X(:,iter)); 	% find the standard deviation of feature for all samples  
    	X(:,iter)= (X(:,iter)-mu(:,iter))/sigma(:,iter);
    end
    % Add intercept term to X
    X = [ones(m, 1) X];


    % Step 2 - Gradient Descent
    % --------------------------
    alpha = 0.01;                       % learning rate
    num_iters = 30000;                  % number of iterations to run gradient descent
    theta = zeros(size(X,2), 1);        % initial value for theta to start from
    J_history = zeros(num_iters, 1);
    precission = 10;                    % How many digits precise the theta value should be

    % no. of iterations determining termination criteria
    for iter = 1:num_iters
        temp_theta = zeros(size(theta));
        for theta_itr = 1:length(theta)
            temp_theta(theta_itr) = theta(theta_itr) - alpha /m * (((X * theta) - y)' * X(:,theta_itr));
        end
        % Temination criteria can also be based on precession
        if isequal( round(theta.*(10^precission)), round(temp_theta.*(10^precission)) )
            J_history = J_history(1:iter-1, 1);
            break;
        end
        theta = temp_theta;
        J_history(iter) = 1/(2 * m) * ((X * theta) - y)' * ((X * theta) - y); % cost for this theta
    end

    fprintf("\nconverged - terminated at iteration = %f\n", iter);

    % Plot the convergence graph
    plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
    xlabel('Number of iterations');
    ylabel('Cost J');

end