import matplotlib.pyplot as plt
import numpy as np


def plot2DData_1(X, symbolValues):
    """
    A private function  to plot 2D data present in X
    Input:
        X:
            The data containing n rows and 2 columns
        symbolValues:
            The dataset can contain different symbols and this numpy array
            (n x 1) contains the symbols in parrallel for each
    don't call from outside
    """
    symbols = np.unique(symbolValues)
    symbolValues = symbolValues.reshape((symbolValues.shape[0],))
    for symbol in np.nditer(symbols):
        samples = X[symbolValues == symbol]
        x1 = samples[:, 0]
        x2 = samples[:, 1]
        plt.plot(x1, x2, "o")


def plot2DData(X, symbolValues=None):
    """
    The function plots 2D data present in X
    Input:
        X:
            The data containing n rows and 2 columns
        symbolValues:
            The dataset can contain different symbols and this numpy array
            (n x 1) contains the symbols in parrallel for each
    """
    if symbolValues is None:
        x1 = X[:, 0]
        x2 = X[:, 1]
        plt.plot(x1, x2, "bo")
    else:
        plot2DData_1(X, symbolValues)
