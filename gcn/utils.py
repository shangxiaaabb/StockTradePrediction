"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import torch

    
    
def unique_rows(data):
    """ Filters unique rows from a 2D np array and also returns inverse indices. Used for edge feature compaction. """
    # https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    uniq, indices = np.unique(data.view(data.dtype.descr * data.shape[1]), return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, data.shape[1]), indices
    
def one_hot_discretization(feat, clip_min, clip_max, upweight):
    indices = np.clip(np.round(feat), clip_min, clip_max).astype(int).reshape((-1,))
    onehot = np.zeros((feat.shape[0], clip_max - clip_min + 1))
    onehot[np.arange(onehot.shape[0]), indices] = onehot.shape[1] if upweight else 1
    return onehot    
    
def get_edge_shards(degs, edge_mem_limit):
    """ Splits iteration over nodes into shards, approximately limited by `edge_mem_limit` edges per shard. 
    Returns a list of pairs indicating how many output nodes and edges to process in each shard."""
    d = degs if isinstance(degs, np.ndarray) else degs.numpy()
    cs = np.cumsum(d)
    cse = cs // edge_mem_limit
    _, cse_i, cse_c = np.unique(cse, return_index=True, return_counts=True)
    
    shards = []
    for b in range(len(cse_i)):
        numd = cse_c[b]
        nume = (cs[-1] if b==len(cse_i)-1 else cs[cse_i[b+1]-1]) - cs[cse_i[b]] + d[cse_i[b]]   
        shards.append( (int(numd), int(nume)) )
    return shards
