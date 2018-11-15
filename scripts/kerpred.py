from K import run as K

import numpy as np

# Input:
#    numpy vector alpha of n rows, 1 column
#    numpy matrix X of features, with n rows (samples), d columns (features)
#        X[i,j] is the j-th feature of the i-th sample
#    numpy vector y of labels, with n rows (samples), 1 column
#        y[i] is the label (+1 or -1) of the i-th sample
#    numpy vector z of d rows, 1 column
#
# Output: label (+1 or -1)

def run(alpha, X, y, z):

    # the number of vectors given as training points
    n = X.shape[0]

    sum = 0
    for i in range(n):
        sum += alpha[i, 0] * y[i, 0] * K(np.array([X[i]]).T, z)

    if sum > 0:
        return +1
    else:
        return -1
