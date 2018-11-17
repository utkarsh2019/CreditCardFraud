import sys
import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm

<<<<<<< HEAD
=======

import linpred as pred
import linprimalsvm as own_svm
>>>>>>> b26917fb0d5d985fa7521139ba75657faff8aa4f
import myopicfitting as mf
import getData as gd
import kFoldCV


F = 15
FOLDS = 1

def test_MF(filename):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    S,thetaS = mf.run(F,X,y)
    z = np.zeros((n,1))
    for i in range(n):
        z[i,0] = np.transpose(thetaS).dot(X[i,S])

    z[z <= 0] = -1
    z[z > 0 ] = 1

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
    if (type == "dual"):
        z = kFoldCV.run(FOLDS, X, y, 1)
    elif (type == "linear"):
        z = kFoldCV.run(FOLDS, X, y, 2)
    else:
        print("Wrong SVM type")
        return
    print("Error: ",(z.mean()))

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Usage: python test.py <input_file> <svm_type>")
    else:
<<<<<<< HEAD
        test_K_Folds_CV(sys.argv[1],sys.argv[2])
=======
        # S = test_MF(sys.argv[1])
        # test_Own_Primal_Svm(sys.argv[1], S)
        # test_Sklearn_Svm(sys.argv[1])
        test_K_Folds_CV(sys.argv[1])
>>>>>>> b26917fb0d5d985fa7521139ba75657faff8aa4f
