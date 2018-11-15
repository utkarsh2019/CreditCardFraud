import numpy as np
import numpy.linalg as la

# Input: numpy matrix X of features, with n rows (samples), d columns (features)
#           X[i,j] is the j-th feature of the i-th sample
#       numpy vector y of scalar values, with n rows (samples), 1 column
#           y[i] is the scalar value of the i-th sample
# Output: numpy vector theta, with d rows, 1 column
# Example on how to call the function:
#   import linreg
#   theta = linreg.run(X,y)

def run(X,y):
    return np.dot(la.pinv(X),y)
