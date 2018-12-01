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
import sys
import numpy as np

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm

import myopicfitting as mf
import getData as gd
import kfoldcv
import pcalearn
import pcaproj
import myPlotHelp

# hyperparameters

# number o features for myopic fitting (feature selection)
F = 15

# number of folds (k) for the k fold cross validation
FOLDS = 5

# slack penalty for svm
C = 0.1

# gamma for radial basis kernel or otherwiae
GAMMA = 0.01

X = None
y = None


def run():
    X, y = gd.getXY()
    
    # print details about X
    print(type(X))
    print(X.shape)
    print()
    
    # print details about y
    print(type(y))
    print(y.shape)
    print()


def makeFeatureNumberGraph(X, y, rangeF, k):
    """
    This graph function visualizes the number of features vs the error

    Input:
        X:
            The samples in shape samples x features

        y:
            The labes in no. samples x 1 form

        rangeF = (startF, stopF):
            F denotes the feature we want to have. rangeF is the range of
            values of F ( the x range in the graph) for which we want to have
            errors. The last value is exclusive (stopF)!

        k: folds in k fold cross validation
    """
    # get a python range object
    rangeF = range(*rangeF)
    xValues = np.arange(rangeF.start, rangeF.stop)
    yValues = []
    for F in rangeF:
        z = kfoldcv.runForFeatureSelection(F, X, y, k)
        meanError = np.mean(z)
        yValues.append(meanError)

    plt.plot(xValues, yValues, "b-")
    plt.xlabel("F")
    plt.ylabel("error")
    plt.show()


def makePCAGraph(X, y):
    """
    This function is to be used for making 2 D graph with PCA
    This is referenced in the report
    """
    mu, Z = pcalearn.run(F=2, X=X)
    projSamples = pcaproj.run(X, mu, Z)

    # projSamples, the projected samples, are now in n x 2 matrix with their
    # projection values for each vector in each row's two columns
    
    # TODO: check if this labels match usage in final report
    myPlotHelp.plot2DData(projSamples, y,
                          {-1: "fraudulent",
                           1: "non-fraudulent"})
    plt.xlabel("PCA vector 1 projection")
    plt.ylabel("PCA vector 2 projection")

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
    


def test_MF(filename):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    S,thetaS = mf.run(F, X, y)
    z = np.zeros((n,1))
    for i in range(n):
        z[i,0] = np.transpose(thetaS).dot(X[i,S])

    z[z <= 0] = -1
    z[z > 0 ] = 1

    print("k fold on myopic with", F, "features")
    matched = z[z == 1]
    score = len(matched) / len(z) * 100
    print("score:", score)
    # plt = matplotlib.pyplot
    # plt.plot(y,"b+")
    # plt.plot(z, "gx")
    print("flag1")
    # plt.show()
    return S

def test_K_Folds_CV(filename,type):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    y = y.reshape(n,)
    if type == "dual":
        z = kfoldcv.run(
                FOLDS, 
                X, 
                y, 
                type="dual", 
                kernel="rbf", 
                gamma=GAMMA,
                C=C)
    elif type == "primal":
        z = kfoldcv.run(
                FOLDS, 
                X, 
                y, 
                type="primal", 
                C=C)
    else:
        print("Wrong SVM type")
        return
    print("Error: ",(z.mean()))

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Usage: python test.py <input_file> <svm_type>")
    else:
        print("details")
        # hyperparameters
        
        # number o features for myopic fitting (feature selection)
        print("F", F)
        
        # number of folds (k) for the k fold cross validation
        print("k", FOLDS)
        
        # slack penalty for svm
        print("C", C)
        
        # gamma for radial basis kernel or otherwiae
        print("gamma", GAMMA)
        
        test_K_Folds_CV(sys.argv[1],sys.argv[2])
        test_MF(sys.argv[1])