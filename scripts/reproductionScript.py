"""
This script can be run to reproduce the experiment that is to be performed
The program will read from the csv containing 1000 samples with roughly 500
500 samples in each of the two classes
"""

import pandas as pd
import numpy as np

from K import run as kernelFunction
import kerdualsvm
import kerpred
import kfoldcv

#CONSTANTS

# the size of k for k fold cross validation
FOLD_SIZE = 5

# the input file
FILE_NAME = "../dataset/creditCardFraudSubset.csv"

if __name__ == "__main__":
    
    # read in the data set
    dataFrame1 = pd.read_csv(FILE_NAME)

    # form X and y

    X = dataFrame1[dataFrame1.columns[:-1]]

    # getting numpy outlook
    X = X.values

    y = dataFrame1["Class"]

    # getting numpy outlook
    y = y.values

    # reshaping
    y = y.reshape((y.shape[0],1))
    # let us decide if we want to analyze it a bit
    
    y[y == 0] = -1
    
    userChoice = input("Do you want to see some descriptive graphs before"
                       " any learning occurs (y/n)")
    if userChoice == "y":
        # let us analyze it a bit
        import matplotlib.pyplot as plt

        import myopicfitting

        # select best two features

    # lets measure on the whole dataset
    alpha = kerdualsvm.run(X, y)
    performance = kfoldcv.measurePerformance(alpha, X, y, X, y)
    print("performance on whole", performance)

    z = kfoldcv.run(5, X, y)
    print("mean:", z.mean())
    n, d = X.shape

