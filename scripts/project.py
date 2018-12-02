"""This is the file with top level functions that can be used in conjunction
with the report

Some functions are not directly used in learning, but help in other things,
such as plotting of graphs. They can be executed independently and are
referenced in the report

To run these functions, perform the following steps
1. run a python console (by typing Python or otherwise)
2. Import project from this package (this file)
3. Execute functions

Enjoy!
"""


# allow the code to be run in Python 2
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

import kfoldcv
import pcalearn
import pcaproj
import myPlotHelp


def makePCAGraph(X, y):
    """
    This function is to be used for making 2 D graph with PCA
    This is referenced in the report
    """
    mu, Z = pcalearn.run(F=2, X=X)
    projSamples = pcaproj.run(X, mu, Z)

    # projSamples, the projected samples, are now in n x 2 matrix with their
    # projection values for each vector in each row's two columns

    myPlotHelp.plot2DData(projSamples, y,
                          {-1: "fraudulent",
                           1: "non-fraudulent"})
    plt.xlabel("PCA vector 1 projection")
    plt.ylabel("PCA vector 2 projection")
    plt.show()


def test_linear_svm(X, y, C, folds):
    """measure performance of linear svm

    Input:
        X:
            input data in n rows and d columns denoting n samples and d
            features. It is a numpy array
        y:
            the labels in n x 1 numpy array
        C:
            The slack variable to adjust in linear svm. It is a floating point
            value
        folds:
            the number of folds in k-folf cross validation. It is an integer
    """
    # error results stored in z, containing error for each fold
    z = kfoldcv.run(folds, X, y, "primal", C=C)
    print("error:", np.mean(z))
    print("stdev:", np.std(z))


def test_non_linear_svm(X, y, C, gamma, folds):
    """measure performance of dual svm

    Input:
        X:
            input data in n rows and d columns denoting n samples and d
            features. It is a numpy array
        y:
            the labels in n x 1 numpy array
        C:
            The slack variable to adjust in linear svm. It is a floating point
            value
        gamma:
            the gamma for the radial basis kernel that we are using. It is a
            floating point value
        folds:
            the number of folds in k-folf cross validation. It is an integer
    """
    # error results stored in z, containing error for each fold
    z = kfoldcv.run(folds, X, y, "dual", C=C, gamma=gamma)
    print("error:", np.mean(z))
    print("stdev:", np.std(z))


def training_testing(X, y, testSetX, testSetY, algorithmType, C, gamma=None):
    y = y.reshape((len(y), ))
    # learn
    if(algorithmType == "primal"):
        clf = svm.LinearSVC(C=C, dual=False)
    elif algorithmType == "dual":
        clf = svm.SVC(kernel="rbf", gamma=gamma, C=C)
    clf.fit(X, y)

    # check performance on testing set
    tempY = testSetY.reshape( (len(testSetY), ) )
    error = 1 - clf.score(testSetX, tempY)
    print("error:", error)
    yPred = clf.predict(testSetX)
    data = kfoldcv.perf_measure(tempY, yPred)
    print("performance:")
    print("TP:", data[0], "FP:", data[1], "TN:", data[2], "FP:", data[3])


def test_K_Folds_CV(X, y, k, algorithmType, C, gamma=None):
    n = X.shape[0]
    y = y.reshape(n,)
    if algorithmType == "dual":
        z = kfoldcv.run(
                k,
                X,
                y,
                algorithmType,
                kernel="rbf",
                gamma=gamma,
                C=C)
    elif algorithmType == "primal":
        z = kfoldcv.run(
                k,
                X,
                y,
                algorithmType,
                C=C)
    else:
        print("Wrong SVM type")
        return
    print("Error: ", (z.mean()))


def calc_stats(X, y, algorithmType, C, gamma=None):
    """
    This function is for obtaining different statistics related to accuracy

    The TP, TN, etc are printed
    """

    n, d = X.shape
    y = y.reshape((n, ))
    # learn
    if(algorithmType == "primal"):
        clf = svm.LinearSVC(C=C, dual=False)
    elif algorithmType == "dual":
        clf = svm.SVC(kernel="rbf", gamma=gamma, C=C)
    clf.fit(X, y)

    # predict
    yhat = clf.predict(X)

    # perf_measure function
    (TP, FP, TN, FN) = kfoldcv.perf_measure(y, yhat)

    # calculate accuracy, precision, sensitivty...
    accuracy = (TP + TN)*1.0/(TP+FP+FN+TN)
    precision = (TP*1.0)/(TP+FP)
    recall = (TP*1.0)/(TP+FN)
    f1_score = 2*(recall * precision) / (recall + precision)

    # return results
    print("TP: ", TP, "\nFP: ", FP, "\nTN: ", TN, "\nFN: ", FN, "\nAccuracy: ", accuracy, "\nPrecision: ", precision, "\nRecall: ", recall, "\nF1 Score: ", f1_score)
