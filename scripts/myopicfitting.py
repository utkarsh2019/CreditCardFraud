import numpy as np
import linreg as lr

# Input: number of features F
#       numpy matrix X of features, with n rows (samples), d columns (features)
#           X[i,j] is the j-th feature of the i-th sample
#       numpy vector y of scalar values, with n rows (samples), 1 column
#           y[i] is the scalar value of the i-th sample
# Output: numpy vector of selected features S, with F rows, 1 column
#       numpy vector thetaS, with F rows, 1 column
#           thetaS[0] corresponds to the weight of feature S[0]
#           thetaS[1] corresponds to the weight of feature S[1]
#           and so on and so forth

def run(F,X,y):
    n,d = X.shape
    S = np.zeros((0,),dtype = np.int32)
    thetaS = np.zeros((0,))
    for f in range(F):
        z = np.zeros((n,1))
        for t in range(n):
            z[t,0] = y[t,0] - np.transpose(thetaS).dot(X[t,S])
        j = [[i] for i in range(d) if i not in S]
        DJ = {}
        for i in j:
            summation = 0
            for t in range(n):
                summation += (z[t,0]*X[t,i])
            DJ[i[0]] = abs(-summation)
        jCap = [max(DJ.keys(), key = DJ.get)]
        thetajCap = lr.run(X[:,jCap],z)
        thetaS = np.concatenate((thetaS,thetajCap.flatten()))
        S = np.concatenate((S,jCap))
    thetaS = np.transpose([thetaS])
    S = np.transpose([S])
    return (S, thetaS)