import sys
import numpy as np

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm

import myopicfitting as mf
import getData as gd
import kfoldcv

# hyperparameters

# number o features for myopic fitting (feature selection)
F = 15

# number of folds (k) for the k fold cross validation
FOLDS = 5

# slack penalty for svm
C = 0.1

# gamma for radial basis kernel or otherwiae
GAMMA = 0.01


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
