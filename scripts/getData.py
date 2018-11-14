import pandas as pd
import csv

def getXY(filename):
    data = pd.read_csv(filename, sep=',',header=None)
    
    X = data[data.columns[0:29]].as_matrix()[1:]
    y = data[data.columns[30]].as_matrix()[1:]

    print("X: ", X)
    print("y: ", y)

    with open('../dataset/X.csv', "wb") as f: 
        writer = csv.writer(f)
        writer.writerows(X)

    with open('../dataset/y.csv', "wb") as f: 
        writer = csv.writer(f)
        writer.writerows(y)
    