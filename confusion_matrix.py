import os
import sys
import glob
import numpy as np
import pickle
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from dassl.utils import set_random_seed


HPARAM_TUPLES = assert(False) #KEVIN


#gts should be npy array with +1 and -1, (N, num_classes)
#pred_probs should be npy array with probabilities, (N, num_classes)
#condition_type should be +1 or -1
#if there are less than count_threshold instances of the thing in the cell, then we'll put np.nan
#will return (num_classes, num_classes) npy array full of probabilities or np.nan
#condition_type==1:  Pr(predict_classB | classA=True, classB=True)
#condition_type==-1: Pr(predict_classB | classA=True, classB=False)
def compute_confusion_matrix(gts, pred_probs, condition_type, count_threshold=100):
    assert(condition_type in [1, -1])
    assert(np.all((gts == 1) | (gts == -1)))
    assert(len(gts.shape) == 2)
    assert(gts.shape == pred_probs.shape)
    assert(np.amin(pred_probs) >= 0)
    assert(np.amax(pred_probs) <= 1)
    row_masks = (gts > 0).astype('int32')
    col_masks = (gts > 0).astype('int32') if condition_type > 0 else (gts < 0).astype('int32')
    counts = row_masks.T @ col_masks #just imagine doing an outer product to each row of row_masks vs corresponding row of col_masks, then summing those outer products together. That's all this is.
    col_probs = probs * col_masks #only count a probB towards columnB if columnB is in mask
    sums = row_masks.T @ col_probs #and only count it towards rowA when rowA is in mask
    out = counts / np.clip(counts, 1e-5, None)
    out[counts < count_threshold] = np.nan
    return out


def get_plot_filename(hparam_tuple, prob_type, epoch):
    assert(False) #KEVIN


def get_gts_and_pred_probs(hparam_tuple, prob_type, epoch):
    set_random_seed(42)
    
    #load model

    #get dataloader

    #run predictions, or grab pseudolabels from pkl file

    #remember to torch.sigmoid()

    #accumulate into big arrays

    #return
    
    assert(False) #KEVIN


def make_plot(confusion_matrix_pos, confusion_matrix_neg, plot_filename):
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    assert(False) #KEVIN


#use this for plotting confusion matrix of ONE checkpoint of ONE experiment for ONE of (test_pred, train_pred, train_pseudo)
#in practice it'll be the two matrices (both condition_type's) stitched together horizontally
#epoch will just have to be a number, i.e. there's no "init" epoch, you'd want to just look at zsclip for that anyway
def confusion_matrix_one(hparam_tuple, prob_type, epoch):
    assert(prob_type in ['test_pred', 'train_pred', 'train_pseudo'])
    plot_filename = get_plot_filename(hparam_tuple, prob_type, epoch)
    gts, pred_probs = get_gts_and_pred_probs(hparam_tuple, prob_type, epoch)
    confusion_matrix_pos = compute_confusion_matrix(gts, pred_probs, 1)
    confusion_matrix_neg = compute_confusion_matrix(gts, pred_probs, -1)
    make_plot(confusion_matrix_pos, confusion_matrix_neg, plot_filename)


def confusion_matrix():
    for hparam_tuple in HPARAM_TUPLES:
        for prob_type in ['test_pred', 'train_pred', 'train_pseudo']:
            for epoch in range(1,51):
                confusion_matrix_one(hparam_tuple, prob_type, epoch)


if __name__ == '__main__':
    confusion_matrix()
