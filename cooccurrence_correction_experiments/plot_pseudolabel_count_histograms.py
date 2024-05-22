import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT


NUM_THRESHOLDS = 25
THRESHOLDS = np.linspace(0, 0.5, NUM_THRESHOLDS + 1)[1:]
NUM_BINS = 15


def plot_pseudolabel_count_histograms(dataset_name):
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    probs = np.array([1 / (1 + np.exp(-pseudolabel_logits[impath])) for impath in sorted(pseudolabel_logits.keys())])
    hists = []
    for threshold in tqdm(THRESHOLDS):
        counts = np.sum(probs > threshold, axis=1)
        hist = np.bincount(counts, minlength=NUM_BINS+5)[:NUM_BINS]
        hists.append(hist)

    hists = np.array(hists)
    plt.clf()
    plt.imshow(hists, vmin=0, vmax=probs.shape[0])
    plt.colorbar()
    plt.yticks(ticks=range(len(THRESHOLDS)), labels=['%.2f'%(threshold) for threshold in THRESHOLDS])
    plt.xticks(range(NUM_BINS))
    plt.title('%s: each row is a histogram - color is freq'%(dataset_name.split('_')[0]))
    plt.xlabel('num positive pseudolabels')
    plt.ylabel('threshold')
    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_CLIP_pseudolabel_count_histograms.png'%(dataset_name.split('_')[0]))
    plt.clf()


def usage():
    print('Usage: python plot_pseudolabel_count_histograms.py <dataset_name>')


if __name__ == '__main__':
    plot_pseudolabel_count_histograms(*(sys.argv[1:]))
