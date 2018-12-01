"""
This file contains the code for a function to measure perfomance of
classification algorithms, given a training set and the prediction function,
and hence the model previously learned
"""

def run(X, y, pred):
    """
    Get a measure of performance on this set with this model
    
    Input:
        X:
            The testing data in a numpy array with n samples, one in each row,
            and d fetures in each column.
        y:
            The true labels corresponding to each sample in X. This is a n x 1
            numpy matrix
        pred:
            This is a callable function that takes a vector and returns the
            label predicted
    """
    n, d = X.shape
    
    for row in range(n):
        
    