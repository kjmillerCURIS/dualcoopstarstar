import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from harvest_training_gts import get_data_manager
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT


NUM_EXAMPLE_CLASSES = 5


def make_pseudolabel_histograms(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    probs = np.array([1 / (1 + np.exp(-pseudolabel_logits[impath])) for impath in sorted(pseudolabel_logits.keys())])
    random.seed(0)
    classes = random.sample(range(len(classnames)), NUM_EXAMPLE_CLASSES)
    plt.clf()
    _, axs = plt.subplots(NUM_EXAMPLE_CLASSES + 1, 2, figsize=(2 * 6.4, (NUM_EXAMPLE_CLASSES + 1) * 4.8))
    for my_class, axrow in zip(classes, axs): #this will take the shorter list, which is classes
        second_flag = False
        for ax in axrow:
            if second_flag:
                ax.hist([p for p in probs[:,my_class] if p > 0.1])
                ax.set_xlim((0.1,1))
            else:
                ax.hist(probs[:,my_class])
                ax.set_xlim((0,1))

            ax.set_xlabel('prob')
            ax.set_ylabel('freq')
            ax.set_title('%s: CLIP pseudolabel probs for "%s"'%(dataset_name.split('_')[0], classnames[my_class]))
            second_flag = True

    second_flag = False
    for ax in axs[-1]:
        if second_flag:
            ax.hist([p for p in probs.flatten() if p > 0.1])
            ax.set_xlim((0.1,1))
        else:
            ax.hist(probs.flatten())
            ax.set_xlim((0,1))

        ax.set_xlabel('prob')
        ax.set_ylabel('freq')
        ax.set_title('%s: CLIP pseudolabel probs for all classes'%(dataset_name.split('_')[0]))
        second_flag = True

    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_CLIP_pseudolabel_probs_histogram.png'%(dataset_name.split('_')[0]))
    plt.clf()


def usage():
    print('Usage: python make_pseudolabel_histograms.py <dataset_name>')


if __name__ == '__main__':
    make_pseudolabel_histograms(*(sys.argv[1:]))
