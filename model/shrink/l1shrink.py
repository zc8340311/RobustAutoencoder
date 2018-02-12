import numpy as np

def shrink(epsilon, x):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou

    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(x*0.)

    for i in xrange(len(x)):
        if x[i] > epsilon:
            output[i] = x[i] - epsilon
        elif x[i] < -epsilon:
            output[i] = x[i] + epsilon
        else:
            output[i] = 0
    return output