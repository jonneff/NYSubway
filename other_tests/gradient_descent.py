import numpy
import pandas

def compute_cost(features, values, theta):
    """
    Compute the cost of a list of parameters, theta, given a list of features 
    (input data points) and values (output data points).
    """
    m = len(values)
    sum_of_square_errors = numpy.square(numpy.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """

    # Perform num_iterations updates to the elements of theta.
    # Every time you compute the cost for a given list of thetas, append it 
    # to cost_history.
    
    cost_history = []

    aom = alpha/len(features)
    for i in range(num_iterations):
        J = compute_cost(features, values, theta)
        h = numpy.dot(features, theta)
        ymh= numpy.subtract (values, h)
        sigma = numpy.dot(ymh, features)
        theta = theta + numpy.dot(aom, sigma)
        cost_history.append(J)

    return theta, pandas.Series(cost_history)
