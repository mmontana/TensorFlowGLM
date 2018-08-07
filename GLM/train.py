import numpy as np
import pandas as pd
import random

def minibatch(indices, size=100, shuffle=True):
    """
    returns a minibatch from indices
    
    :param indices: list or range of indices
    :param size: size of the batch
    :param shuffle: random shuffling the data
    :type indices: list
    :type size: int, float in open interval (0,1)
    :type shuffle: Boolean
    :return: minibatch indices and remaining indices in pool
    :rtype: list, list
    """
    
    if type(indices) is range:
        indices = list(indices)
        
    n_samples = len(indices)
        
    if size < 1 and size > 0:
        size = np.floor(size*n_samples).astype(int)
    if size >= n_samples:
        return indices, None
    
    if shuffle:
        random.shuffle(indices)
    
    return indices[:size], indices[size:]

def epoch():
    pass





