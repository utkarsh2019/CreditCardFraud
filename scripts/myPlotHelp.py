import matplotlib.pyplot as plt
import numpy as np


def plot2DData_1(X, symbolValues, labels):
    """
    A private function  to plot 2D data present in X
    Input:
        X:
            The data containing n rows and 2 columns

        symbolValues:
            The dataset can contain different symbols and this numpy array
            (n x 1) contains the symbols in parrallel for each

        labels:
            A dict (dictionary) object containing mapping from symbol value to
            a label for that symbol (such as fraudulent for -1)
    don't call from outside
    """
    symbols = np.unique(symbolValues)
    symbolValues = symbolValues.reshape((symbolValues.shape[0],))
    for symbol in np.nditer(symbols):
        # getting the only value inside this array. Numpy returns an array for
        # symbol
        symbol = symbol.item()
        samples = X[symbolValues == symbol]
        x1 = samples[:, 0]
        x2 = samples[:, 1]
        
        if labels is not None:
            plt.plot(x1, x2, ".", label=labels[symbol])
        else:
            plt.plot(x1, x2, ".")
    
    plt.legend()


def plot2DData(X, symbolValues=None, labels=None):
    """
    The function plots 2D data present in X
    Input:
        X:
            The data containing n rows and 2 columns

        symbolValues:
            The dataset can contain different symbols and this numpy array
            (n x 1) contains the symbols in parrallel for each

        labels:
            A dict (dictionary) object containing mapping from symbol value to
            a label for that symbol (such as fraudulent for -1)
    """
    if X.shape[1] != 2:
        raise ValueError("The dimension of data to plot does not contain 2" +
                         "columns")
    if symbolValues is None:
        x1 = X[:, 0]
        x2 = X[:, 1]
        plt.plot(x1, x2, "bo")
    else:
        plot2DData_1(X, symbolValues, labels)
