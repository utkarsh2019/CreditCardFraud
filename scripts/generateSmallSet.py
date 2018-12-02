"""
This executable is for creating a subset of input data with the same
proportions of classes
"""
import sys
import pandas as pd
import numpy as np
import math

OUTPUT_SIZE = 5000

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("wrong input arguments")
    inCSV = sys.argv[1]
    outCSV = sys.argv[2]

    inputDf = pd.read_csv(inCSV)
    y = inputDf[inputDf.columns[-1]]

    inPostiveCount = len(y[y == 1])
    inNegativeCount = len(y[y == -1])

    positiveCount = (inPostiveCount * 1.0 / (inPostiveCount + inNegativeCount) 
                        * OUTPUT_SIZE )
    positiveCount = math.floor(positiveCount)
    negativeCount = OUTPUT_SIZE - positiveCount

    print(positiveCount, negativeCount)
    positiveSet = inputDf[y == 1]
    negativeSet = inputDf[y == -1]

    sampleSet1 = positiveSet.sample(n=positiveCount)
    sampleSet2 = negativeSet.sample(n=negativeCount)
    dftemp = pd.DataFrame(sampleSet1)
    outputDf = dftemp.append(sampleSet2)
    outputDf.to_csv(outCSV)

