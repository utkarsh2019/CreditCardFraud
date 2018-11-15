import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import linpred as pred
import linprimalsvm as svm
import myopicfitting as mf
import getData as gd

F = 16

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

def test_Svm(filename, S):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    print(S[:,0].shape)
    X1 = X[:n//2,S[:,0]]
    print(X1.shape)
    y1 = y[:n//2]
    X2 = X[n//2:,S[:,0]]
    print(X2.shape)
    y2 = y[n//2:]
    theta = svm.run(X1,y1)
    test_Pred(theta,X2,y2)

def test_MF(filename):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    S,thetaS = mf.run(F,X,y)
    # print S
    z = np.zeros((n,1))
    for i in xrange(n):
        z[i,0] = np.transpose(thetaS).dot(X[i,S])
    #for i in range(n):
    #    print abs(y[i,0] - z[i,0])
    # plt = matplotlib.pyplot
    # plt.plot(y,"b")
    # plt.plot(z)
    print("flag1")
    # plt.show()
    return S

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python test.py <input_file>")
    else:
        S = test_MF(sys.argv[1])
        test_Svm(sys.argv[1], S)