import numpy as np

numberOfColumns = 31
defaultRows = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

def getDataFrom(filepath, rowsToSelect=defaultRows, n=1000, withHeaders=True):
    """
    Converts an input .csv file into a numpy array 

    Parameters
    ----------
    filepath - path to the input .csv file
    rowsToSelect - a list of the row indices to select
    n - number of data rows in the file
    withHeaders - if the header line is present in the file

    Returns
    -------
    npData - numpy array of the select columns of data read from the file

    Examples
    --------
    data = getDataFrom('./../dataset/creditCardFraudSubset.csv', rowsToSelect=[0,1,30], n=1000)
    """
    npData = np.zeros((n,len(rowsToSelect)))
    lineCounter = -1

    infile = open(filepath, 'r')
    if withHeaders:
        # Ignore the header line in the file
        infile.readline()
    try:
        for line in infile:
            lineCounter = lineCounter + 1
            tokens = line.split(',')
            for i in range(len(rowsToSelect)):
                try:
                    npData[lineCounter][i] = tokens[rowsToSelect[i]]
                except:
                    if tokens[rowsToSelect[i]] == '"1"\n':
                        npData[lineCounter][i] = 1
                    elif tokens[rowsToSelect[i]] == '"0"\n':
                        npData[lineCounter][i] = 0
                    else:
                        npData[lineCounter][i] = -1
    except IndexError:
        pass
    infile.close()

    return npData
    