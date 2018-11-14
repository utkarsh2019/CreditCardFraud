# import createsepdata as csd
# import linperceptron as perc
import linpred as pred
import linprimalsvm as svm
import numpy as np
import getData as gd

# def test_Perc():
#     X,y = csd.run(100,3)
#     theta = perc.run(5,X,y)
#     print(theta)
#     test_Pred(theta,X,y)

def test_Pred(theta,X,y):
    Z = np.zeros((X.shape[0],1))
    count = 0
    for i in range(250):
        Z[i,0] = pred.run(theta,np.transpose(np.array([X[i,:]])))
        if y[i,0] == Z[i,0]:
            count += 1
    print(count)
    print('Yay')
    return 1

def test_Svm():
    X,y = gd.getXY('../dataset/250.csv')
    print(X)
    print(y)
    theta = svm.run(X,y)
    print(theta)
    test_Pred(theta,X,y)

if __name__ == "__main__":
    test_Svm()