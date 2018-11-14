import numpy as np
import cvxopt as co

# Input: numpy matrix X of features, with n rows (samples), d columns (features)
#       X[i,j] is the j-th feature of the i-th sample
#   numpy vector y of labels, with n rows (samples), 1 column
#       y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column

def run(X,y):
    H = np.identity(X.shape[1])
    f = np.zeros((X.shape[1],1))
    A = np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            A[i][j] = -y[i]*X[i][j]
    b = np.ones((X.shape[0],1))
    b[:] = -1
    theta = np.array(co.solvers.qp(co.matrix(H,tc='d'),co.matrix(f,tc='d'),co.matrix(A,tc='d'),co.matrix(b,tc='d'))['x'])
    return theta