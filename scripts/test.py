import sys
import numpy as np

from sklearn import svm

import linpred as pred
import linprimalsvm as own_svm
import myopicfitting as mf
import getData as gd
import kfoldcv

F = 15
FOLDS = 5

def percentage(part, whole):
  return 100 * float(part)/float(whole)

def test_Pred(theta,X,y):
    n = X.shape[1]
    count = 0
    for i in range(n):
        if y[i,0] == pred.run(theta,np.transpose(np.array([X[i,:]]))):
            count += 1
    print("Error: " + str(percentage(n-count,n)) + "%")
    return 1

def test_Own_Primal_Svm(filename, S):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    X1 = X[:n//2,S[:,0]]
    y1 = y[:n//2]
    X2 = X[n//2:,S[:,0]]
    y2 = y[n//2:]
    theta = own_svm.run(X1,y1)
    test_Pred(theta,X2,y2)

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

def test_Sklearn_Svm(filename):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    X1 = X[:n//2]
    y1 = y[:n//2].reshape(n//2,)
    X2 = X[n//2:]
    y2 = y[n//2:].reshape(n//2,)
    clf = svm.SVC(kernel='linear',C=1,gamma=1)
    clf.fit(X1,y1)
    count = 0
    n1 = n//2
    for i in range(n1):
        if y2[i] == clf.predict([X2[i]])[0]:
            count += 1
    print("Error: " + str(percentage(n1-count,n1)) + "%")

def test_K_Folds_CV(filename):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    y = y.reshape(n,)
    z = kfoldcv.run(FOLDS, X, y, 1)
    print("Error: ",(z.mean()*100))

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python test.py <input_file>")
    else:
        # S = test_MF(sys.argv[1])
        # test_Own_Primal_Svm(sys.argv[1], S)
        # test_Sklearn_Svm(sys.argv[1])
        test_K_Folds_CV(sys.argv[1])