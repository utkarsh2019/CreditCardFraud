import pandas as pd
import numpy as np
import csv

def getXY(filename):
    data = pd.read_csv(filename, sep=',',header=None)
    X = data[data.columns[0:29]].values[1:].astype(np.float)
    y = np.transpose([data[data.columns[30]].values[1:].astype(np.float)])
    return X, y

if __name__ == "__main__":
    getXY('../dataset/250.csv')