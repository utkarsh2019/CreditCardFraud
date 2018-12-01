import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import linpred as pred
import linprimalsvm as svm
import myopicfitting as mf
import getData as gd

import test

def plotMyopicForwardFitting():
    # Graphing
    X,y = gd.getXY("./dataset/1000.csv")
    S = test.test_MF("./dataset/1000.csv")
    # Lists to hold points for scatter plot
    xvalsFraud = []
    yvalsFraud = []
    xvalsNormal = []
    yvalsNormal = []
    # Assign points to lists
    for i in range(len(X)):
        if y[i] == 1: # Non-fraudulent
            xvalsNormal += [X[i][S[1]]]
            yvalsNormal += [X[i][S[2]]]
        elif y[i] == -1: # Fraudulent
            xvalsFraud += [X[i][S[1]]]
            yvalsFraud += [X[i][S[2]]]
    
    # Plot scatter plots
    matplotlib.pyplot.scatter(xvalsNormal, yvalsNormal, color = "green", marker = '+', s=30, alpha=1.0)
    matplotlib.pyplot.scatter(xvalsFraud, yvalsFraud, color="red", marker='_', s=30, alpha=1.0)
    # Show the plot
    matplotlib.pyplot.show()

# - - - - - - - - - - #
plotMyopicForwardFitting()
