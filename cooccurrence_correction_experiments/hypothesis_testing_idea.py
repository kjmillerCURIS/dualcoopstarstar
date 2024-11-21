import os
import sys
import numpy as np
from tqdm import tqdm


#z should be of shape (N, M)
#musigma should be of shape (M, 2)
#return something of shape (N, M)
def compute_gauss_probs(z, musigma):
    assert(False) #KEVIN


def compute_y_cond_Xi_helper(clip_scores, yi_cond_Xi_0_musigma, yi_cond_Xi_1_musigma, Xk_cond_Xi):
    #P(y | X_i) = prod_k sum_j P(y_k | X_k=j) P(X_k=j | X_i)
    #no need for special case of k==i, as long as Xk_cond_Xi diagonals are correct
    #we will compute the sums individually, then log, then sum over k, then exp!
    #no, this will not lead to taking a log of 0, because either P(X_k=0 | X_i) or P(X_k=1 | X_i) will be positive, and P(y_k | X_k=j) will always be positive because gaussian
    (N, M) = clip_scores.shape
    gauss_probs_0 = compute_gauss_probs(clip_scores, yi_cond_Xi_0_musigma)
    gauss_probs_1 = compute_gauss_probs(clip_scores, yi_cond_Xi_1_musigma)
    terms_0 = gauss_probs_0[:,np.newaxis,:] * (1 - Xk_cond_Xi)[np.newaxis,:,:]
    terms_1 = gauss_probs_1[:,np.newaxis,:] * Xk_cond_Xi[np.newaxis,:,:]
    assert(terms_0.shape == (N, M, M))
    assert(terms_1.shape == (N, M, M))
    terms = terms_0 + terms_1
    y_cond_Xi = np.exp(np.sum(np.log(terms), axis=-1))
    assert(y_cond_Xi.shape == (N, M))
    return y_cond_Xi


#computes P(y | X_i)
#clip_scores should be shape (N, M) where N is number of datapoints and M is number of classes
#yi_cond_Xi_0_musigma is mean and sd (in that order) of P(y_i | X_i=0), shape is (M, 2)
#Xk_cond_Xi_0[i, k] = P(X_k=1 | X_i=0), shape is (M, M) and EVERY entry should be correct, INCLUDING diags
#returns y_cond_Xi_0 and y_cond_Xi_1, which is P(y | X_i=0) and P(y | X_i=1)
#these are shape (N, M)
def compute_y_cond_Xi(clip_scores, yi_cond_Xi_0_musigma, yi_cond_Xi_1_musigma, Xk_cond_Xi_0, Xk_cond_Xi_1):
    assert(np.all(np.diag(Xk_cond_Xi_0) == 0))
    assert(np.all(np.diag(Xk_cond_Xi_1) == 1))
    y_cond_Xi_0 = compute_y_cond_Xi_helper(clip_scores, yi_cond_Xi_0_musigma, yi_cond_Xi_1_musigma, Xk_cond_Xi_0)
    y_cond_Xi_1 = compute_y_cond_Xi_helper(clip_scores, yi_cond_Xi_0_musigma, yi_cond_Xi_1_musigma, Xk_cond_Xi_1)
    return y_cond_Xi_0, y_cond_Xi_1

    assert(False) #KEVIN
