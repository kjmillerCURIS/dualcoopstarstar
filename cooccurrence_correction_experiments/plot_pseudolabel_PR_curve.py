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


def plot_pseudolabel_PR_curve(dataset_name, classname):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    class_index = classnames.index(classname)
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    assert(sorted(gts.keys()) == sorted(pseudolabel_logits.keys()))
    targets = np.array([gts[impath][class_index] for impath in sorted(gts.keys())])
    scores = np.array([pseudolabel_logits[impath][class_index] for impath in sorted(pseudolabel_logits.keys())])
    scores = 1 / (1 + np.exp(-scores))
    my_AP = average_precision(scores, targets)
    _, axs = plt.subplots(2, 1, figsize=(6.4, 2 * 4.8))
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    axs[0].plot(recall, precision)
    axs[0].set_xlim((0,1))
    axs[0].set_ylim((0,1))
    axs[0].set_xlabel('recall')
    axs[0].set_ylabel('precision')
    axs[0].set_title('%s - %s PR curve, AP = %f'%(dataset_name.split('_')[0], classname, my_AP))
    axs[1].plot(thresholds, recall[:-1], color='r', label='recall')
    axs[1].plot(thresholds, precision[:-1], color='b', label='precision')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('threshold')
    axs[1].set_ylabel('precision / recall')
    axs[1].legend()
    axs[1].set_ylim((0,1))
    axs[1].set_title('precision and recall vs threshold')
    plt.tight_layout()
    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_%s_PR_curve.png'%(dataset_name.split('_')[0], classname), dpi=300)


def usage():
    print('Usage: python plot_pseudolabel_PR_curve.py <dataset_name> <classname>')


if __name__ == '__main__':
    plot_pseudolabel_PR_curve(*(sys.argv[1:]))
