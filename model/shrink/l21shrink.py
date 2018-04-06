import numpy as np

def l21shrink(epsilon, x):
    """
    auther : Chong Zhou
    date : 10/20/2016
    Args:
        epsilon: the shrinkage parameter
        x: matrix to shrink on
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
    Returns:
            The shrunk matrix
    """
    output = x.copy()
    norm = np.linalg.norm(x, ord=2, axis=0)
    for i in xrange(x.shape[1]):
        if norm[i] > epsilon:
            for j in xrange(x.shape[0]):
                output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
        else:
            output[:,i] = 0.
    return output