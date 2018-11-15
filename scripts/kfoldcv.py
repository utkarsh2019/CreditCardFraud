import numpy as np

import kerdualsvm
import kerpred


def measurePerformance(alpha, X, y, X1, y1):
    """
    This feature computes the score of a specific set, with the model learned

    Input:
        alpha: the model learned is denoted by alpha, and the X and y above
        X: The data points from which the model is learnt
        y: the labels for each data point from which the model is learnt
        X1: The data points which are under consideration - for which the score
            is to be computed
        y1: The true labels of the data points X1 that are to be tested

    Output:
        A value between 0 and 1 inclusive, representing the score on this set
    """
    n, d = X1.shape

    nCorrect = 0
    for i in range(n):
        x = X1[i]
        x = x.reshape((d, 1))
        prediction = kerpred.run(alpha, X, y, x)
        if prediction == y1[i, 0]:
            nCorrect += 1

    return nCorrect * 1.0 / n

# Input: number of folds k
#        numpy matrix X of features, with n rows (samples), d columns
#        (features)
#        numpy vector y of labels 1 or 0, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column. Read details below


def run(k, X, y):
    n, d = X.shape

    # this is a list containing all the k subsets. So it is of size k in the
    # once we ready it, in a for loop
    allSets = []

    # there is a parallel list for labels
    ySets = []
    # this loop is not the main loop to iterate and fill z. It is just to make
    # the sets
    for i in range(k):
        # we use float to perform normal division (not integer division)
        lower = int(float(n) * i / float(k))
        upper = int(float(n) * (i+1) / float(k)) - 1
        subsetForX = X[lower:upper+1, ]
        subsetForY = y[lower:upper+1, ]
        allSets.append(subsetForX)
        ySets.append(subsetForY)

    # z contains the resulting mean square error for each set according to k
    # fold cross validation
    z = np.zeros(shape=(k, 1))

    # the main loop matching the pseudo code given in the question
    for i in range(k):
        T = allSets[i]
        yForT = ySets[i]

        # we try and add all rows that are not in T, into setForS
        setForS = allSets[0:i]
        setForS.extend(allSets[i + 1:])

        # we convert rows or sets of rows, into a common array S
        S = np.concatenate(setForS)

        # parralel conversion for making a set of corresponding labels
        yForS = ySets[0:i]
        yForS.extend(ySets[i + 1:])
        yForS_1 = np.concatenate(yForS)

        alpha = kerdualsvm.run(S, yForS_1)

        # getting performance measure of test set
        z[i] = 1 - measurePerformance(alpha, S, yForS_1, T, yForT)

    return z
