import os
import sys
import numpy as np
import pickle
from compute_mAP import average_precision


def standardize_helper(scores, axis, num_classes=None):
    scores_for_stats = scores
    if num_classes is not None:
        assert(axis == 1)
        scores_for_stats = scores[:,:num_classes]

    return (scores - np.mean(scores_for_stats, axis=axis, keepdims=True)) / np.std(scores_for_stats, ddof=1, axis=axis, keepdims=True)


def calibrate(scores, num_classes=None):
    return standardize_helper(standardize_helper(scores, 1, num_classes=num_classes), 0)


def meow(scores_filename_ens80, scores_filename_waffle):
    with open(scores_filename_ens80, 'rb') as f:
        d_ens80 = pickle.load(f)

    with open(scores_filename_waffle, 'rb') as f:
        d_waffle = pickle.load(f)

    num_classes = len(d_ens80['classnames'])
    scores_ens80 = calibrate(d_ens80['cossims'])
    scores_waffle = d_waffle['cossims']
    scores_waffle = calibrate(scores_waffle, num_classes=num_classes)
    scores_waffle = np.reshape(d_waffle['cossims'][:,num_classes:], (d_waffle['cossims'].shape[0], -1, num_classes))
    print(scores_waffle.shape)
    scores_waffle = np.mean(scores_waffle, axis=1)
    print(scores_waffle.shape)
    scores = 0.5 * scores_ens80 + 0.5 * scores_waffle

    APs = [100.0 * average_precision(scores[:,i], d_ens80['gts'][:,i]) for i in range(d_ens80['gts'].shape[1])]
    mAP = np.mean(APs)
    print(mAP)


if __name__ == '__main__':
    meow(*(sys.argv[1:]))
