import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels for each data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d, ) NumPy array containing the weights of linear regression
    """
    n, d = X.shape
    identity_matrix = np.identity(d)

    # Compute the closed-form solution
    theta = np.linalg.inv(X.T @ X + lambda_factor * identity_matrix) @ X.T @ Y
    
    return theta
    raise NotImplementedError

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
