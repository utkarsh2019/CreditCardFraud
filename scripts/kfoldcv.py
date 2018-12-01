import numpy as np
from sklearn import svm

import myopicfitting
import linpred

# Calculate and return true positive, true negative, false positive, and false negative
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


def processAll(X, y, type):
    z = np.zeros(shape=(1, 1))

    # for dual svm using sklearn
    if (type == 1):
        # clf = svm.SVC(kernel='linear',C=1,gamma=1)
        clf = svm.SVC(kernel="rbf", gamma=0.0000001, C=100000)
        clf.fit(X, y)
        count = 0
        lenY = len(y)
        for j in range(lenY):
            if y[j] == clf.predict([X[j]])[0]:
                count += 1
        z[0] = 100*float(lenY - count)/float(lenY)
        print(z[0])

    # for primal svm using sklearn
    elif(type == 2):
        clf = svm.LinearSVC(C=10, dual=False)
        clf.fit(X, y)
        count = 0
        lenY = len(y)
        for j in range(lenY):
            if y[j] == clf.predict([X[j]])[0]:
                count += 1
        z[0] = 100*float(lenY - count)/float(lenY)
        print(z[0])
    return z


def run(k, X, y, algorithmType, **kwargs):
    """
    Perform k fold cross validation and return an array representing the
    results

    Input:
        folds:
            The value of k, the number of folds in k-fold cross validation.

        X:
            The samples, as an n x d numpy array

        y:
            The labels in parallel with the samples above

        Keyword arguments, **kwargs:
            C:
                The slack variable for both primal and dual svm

            algorithmType:
                A python string with value as primal or dual for the
                appropriate model. We are using the radial basis kernel
                in the case of the dual mode

            gamma:
                Read svm.LinearSVC for details. For the radial basis kernel, it
                is the hyperparameter to be tuned

    Return:
        The function would return a matrix z of dimensions k x 1, containing
        the error percentage (%) for each of the k trials.

    """
    # TODO: processAll necc?

    n = X.shape[0]

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

        yForT = yForT.reshape( (yForT.shape[0],) )
        yS = yForS_1.reshape( (yForS_1.shape[0],) )
        # for dual svm using sklearn
        if algorithmType == "primal":
            # clf is the classifier object for performing learning, testing and
            # more
            clf = svm.LinearSVC(C=kwargs["C"], dual=False)
            clf.fit(S, yS)
            # get the error on the set in T
            # (see documentation for details)
            z[i] = 1 - clf.score(T, yForT)

        # for primal svm using sklearn
        elif algorithmType == "dual":
            clf = svm.SVC(
                        kernel="rbf",
                        gamma=kwargs["gamma"],
                        C=kwargs["C"])

            clf.fit(S, yS)
            z[i] = 1 - clf.score(T, yForT)
    return z