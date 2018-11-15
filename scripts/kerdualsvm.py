from K import run as K

import numpy as np
import cvxopt as co

# Input:
#   numpy matrix X of features, with n rows (samples), d columns (features)
#       X[i,j] is the j-th feature of the i-th sample
#   numpy vector y of labels, with n rows (samples), 1 column
#       y[i] is the label (+1 or -1) of the i-th sample
#
# Output: numpy vector alpha of n rows, 1 column

def run(X, y):
    n = X.shape[0]
    d = X.shape[1]

    f = np.full(shape=(n, 1), fill_value=-1)
    b = np.zeros(shape=(n, 1))

    a = np.full(shape=(n, n), fill_value=-1)
    a = np.diag(np.diag(a))

    # initializing H with zeroes
    h = np.zeros(shape=(n, n))

    # filling H with kernel values
    for i in range(n):
        for j in range(n):
            y1 = y[i, 0]
            y2 = y[j, 0]
            x1 = np.array([X[i]]).T
            x2 = np.array([X[j]]).T
            h[i, j] = y1 * y2 * K(x1, x2)

    alpha = np.array(co.solvers.qp(
                        co.matrix(h, tc='d'),
                        co.matrix(f, tc='d'),
                        co.matrix(a, tc='d'),
                        co.matrix(b, tc='d'))['x'])
    return alpha
    