import numpy as np

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    
    ybar = np.mean(data)
    r_squared = 1 - np.sum(np.square(data-predictions)) / np.sum(np.square(data-ybar))

    return r_squared