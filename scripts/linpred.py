"""
This file contains implementaion for the linear predictor.
The run function is to be called by external clients
"""

import numpy as np
# Input: numpy vector theta of d rows, 1 column
#        numpy vector x of d rows, 1 column
# Output: label (+1 or -1)
def run(theta, x):
    check = np.dot(theta.transpose(), x)
    if check <= 0:
        return -1
    else:
        return 1
