import numpy as np

# Input: numpy vector theta of d rows, 1 column
#   numpy vector x of d rows, 1 column
# Output: label (+1 or -1)

def run(theta,x):
    if np.transpose(theta).dot(x) > 0:
        label = 1
    else:
        label = 0
    return label
