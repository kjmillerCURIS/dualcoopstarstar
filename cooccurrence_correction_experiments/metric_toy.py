import os
import sys
import copy
import numpy as np
import random
from tqdm import tqdm


NUM_SAMPLES = 1000000
RHO = 0.5
B = 0.1733333
Q = 0.04


def metric_toy():
    G = np.array([[0.,0.],[0.,1.],[1.,0.]])
    g_indices = np.random.randint(0, high=3, size=NUM_SAMPLES)
    g = G[g_indices]
    noise_mask = (np.random.uniform(size=NUM_SAMPLES) < RHO)
    noise_mask = noise_mask & (g[:,0] != g[:,1])
    z_hat = copy.deepcopy(g)
    z_hat[noise_mask,:] = [0.5, 0.5]
    d_metric = np.mean(g[:,0] * g[:,1]) / (np.mean(g[:,0]) * np.mean(g[:,1]))
    d_hat_metric = np.mean(z_hat[:,0] * z_hat[:,1]) / (np.mean(z_hat[:,0]) * np.mean(z_hat[:,1]))
    d_hat_metric_expected = (np.mean(g[:,0] * g[:,1]) + B + Q) / (np.mean(g[:,0]) * np.mean(g[:,1]) + B)
    print(d_metric)
    print(d_hat_metric)
    print(d_hat_metric_expected)
    print(np.sum((1 - g[:,0]) * z_hat[:,0]) / np.sum(1 - g[:,0]))
    print(np.sum(g[:,0] * z_hat[:,0]) / np.sum(g[:,0]))


if __name__ == '__main__':
    metric_toy()
