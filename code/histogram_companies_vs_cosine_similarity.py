import matplotlib.pyplot as plt
import numpy as np

def arrange_data(data):
    X = [1 - d['cosine_distance'] for d in data]
    return X

def mean(X):
    return np.mean(X)

def std_dev(X):
    return np.std(X)
