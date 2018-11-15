import sys
import linpred as pred
import linprimalsvm as svm
import numpy as np
import getData as gd

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

def test_Svm(filename):
    X,y = gd.getXY(filename)
    n = X.shape[0]
    X1 = X[:n/2]
    y1 = y[:n/2]
    X2 = X[n/2:]
    y2 = y[n/2:]
    theta = svm.run(X1,y1)
    test_Pred(theta,X2,y2)

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python test.py <input_file>")
    else:
        test_Svm(sys.argv[1])