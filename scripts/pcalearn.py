import numpy as np
import numpy.linalg as la

# Input: number of features F
#        numpy matrix X, with n rows (samples), d columns (features)
# Output: numpy vector mu, with d rows, 1 column
#         numpy matrix Z, with d rows, F columns


def run(F, X):
    n, d = X.shape
    mu = np.empty(shape=(d, 1))

    # getting the mu for each dimension
    # mu is d rows, 1 column
    mu = np.mean(X, axis=0).reshape((d, 1))

    # center the data (and still the eigen vectors are alright, or rather they
    # can only be computed like this)

    # there is a broadcast operation to remove mu value for resepctive
    # dimensions
    X = (X.T - mu).T

    U, s, Vt = la.svd(X)

    # take the first F entries in s
    g = s[:F]

    g = np.where(g > 0, 1.0/g, g)

    W = Vt[:F]
    Z = np.matmul(W.T, np.diag(g))
    return mu, Z
