import os
import pickle

import tensorflow as tf
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)

def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

def cal_gcn_matrix(A):
    I = np.diag(np.ones(A.shape[0], dtype=np.float32))
    D_diag = A.sum(axis=1)
    D_ = np.diag(np.power(D_diag, -1/2))
    D_[D_ == np.inf] = 0
    return I + np.matmul(np.matmul(D_, A), D_)
    
def cal_matrix(A):
    D = A.sum(axis=1)
    return tf.convert_to_tensor(np.array([A[i]/s if s>0 else A[i] for i,s in enumerate(D)]),tf.float32)

def xavier_init(shape, constant=1):
    fan_in = shape[-2]
    fan_out = shape[-1]
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)