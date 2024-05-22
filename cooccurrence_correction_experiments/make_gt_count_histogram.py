import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT


def make_gt_count_histogram(dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gt_labels = pickle.load(f)

    assert(all([np.all((gt_labels[impath]==0) | (gt_labels[impath]==1)) for impath in sorted(gt_labels.keys())]))
    counts = [int(np.sum(gt_labels[impath])) for impath in sorted(gt_labels.keys())]
    bins = np.arange(0, np.amax(counts) + 2) - 0.5  # +2 to include the last value in the bin range
    plt.clf()
    plt.hist(counts, bins=bins, edgecolor='black')
    plt.xticks(range(0, np.amax(counts) + 2))
    plt.title('%s: gt label # positives per image histogram'%(dataset_name.split('_')[0]))
    plt.xlabel('# positive labels per image')
    plt.ylabel('freq')
    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_gt_num_positives_per_image_histogram.png'%(dataset_name.split('_')[0]))
    plt.clf()


def usage():
    print('Usage: python make_gt_count_histogram.py <dataset_name>')


if __name__ == '__main__':
    make_gt_count_histogram(*(sys.argv[1:]))
