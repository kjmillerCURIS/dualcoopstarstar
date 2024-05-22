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


def plot_pseudolabel_APs(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    assert(sorted(gts.keys()) == sorted(pseudolabel_logits.keys()))
    targets = np.array([gts[impath] for impath in sorted(gts.keys())])
    scores = np.array([pseudolabel_logits[impath] for impath in sorted(pseudolabel_logits.keys())])
    my_APs = np.array([average_precision(scores[:,i], targets[:,i]) for i in range(scores.shape[1])])

    plt.clf()
    plt.figure(figsize=(20, 10))  # Wide figure to accommodate 80 bars
    plt.bar(classnames, my_APs, color='skyblue')
    plt.ylabel('AP')
    plt.title('%s: CLIP pseudolabel APs'%(dataset_name.split('_')[0]))
    plt.xticks(rotation=90)  # Rotate the animal names to fit them better
    plt.ylim((0,1))
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_individual_APs.png'%(dataset_name.split('_')[0]), dpi=300)
    plt.clf()


def usage():
    print('Usage: python plot_pseudolabel_APs.py <dataset_name>')


if __name__ == '__main__':
    plot_pseudolabel_APs(*(sys.argv[1:]))
