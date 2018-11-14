import pandas as pd
import numpy as np
import csv

def getXY(filename):
    data = pd.read_csv(filename, sep=',',header=None)
    # print(data)
    X = data[data.columns[0:29]].values[1:].astype(np.float)
    y = np.transpose([data[data.columns[30]].values[1:].astype(np.float)])
    # print(X)
    # print(y)
    # print(type(X))
    # print(X.shape)
    # print(type(y))
    # print(y.shape)
    return X,y

if __name__ == "__main__":
    getXY('../dataset/250.csv')