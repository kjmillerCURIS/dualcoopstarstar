import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, get_data_manager
from compute_mAP import average_precision


def plot_marginals(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    assert(sorted(gts.keys()) == sorted(pseudolabel_logits.keys()))
    targets = np.array([gts[impath] for impath in sorted(gts.keys())])
    scores = np.array([pseudolabel_logits[impath] for impath in sorted(pseudolabel_logits.keys())])
    probs = 1 / (1 + np.exp(-scores))
    gt_marginals = np.mean(targets, axis=0)
    pseudo_marginals = np.mean(probs, axis=0)

    plt.clf()
    _, axs = plt.subplots(2, 1, figsize=(20, 20))  # Wide figure to accommodate 80 bars
    axs[0].bar(classnames, gt_marginals, color='skyblue')
    axs[0].set_ylabel('marginal prob')
    axs[0].set_title('%s: gt marginals'%(dataset_name.split('_')[0]))
    axs[0].set_xticks(range(len(classnames)))  # Set x-tick positions
    axs[0].set_xticklabels(classnames, rotation=90)
    axs[0].set_ylim((0,1))
    axs[1].bar(classnames, pseudo_marginals, color='skyblue')
    axs[1].set_ylabel('marginal prob')
    axs[1].set_title('%s: CLIP pseudolabel marginals'%(dataset_name.split('_')[0]))
    axs[1].set_xticks(range(len(classnames)))  # Set x-tick positions
    axs[1].set_xticklabels(classnames, rotation=90)
    axs[1].set_ylim((0,1))
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_marginals.png'%(dataset_name.split('_')[0]), dpi=300)
    plt.clf()


def usage():
    print('Usage: python plot_marginals.py <dataset_name>')


if __name__ == '__main__':
    plot_marginals(*(sys.argv[1:]))
