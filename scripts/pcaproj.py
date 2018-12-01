"""
This file is for using a function for projecting the a set of real data into
PCA space
"""
import numpy as np
# Input:
#        number of features F
#        numpy matrix X, with n rows (samples), d columns (features)
#        numpy vector mu, with d rows, 1 column
#        numpy matrix Z, with d rows, F columns
# Output:
#        numpy matrix P, with n rows, F columns

def run(X, mu, Z):
    # Your code goes here
    X = (X.T - mu).T
    P = np.matmul(X, Z)
    return P
