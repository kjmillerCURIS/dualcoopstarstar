import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm


def make_qualitative_histograms(extra_filename, plot_filename):
    with open(extra_filename, 'rb') as f:
        extra = pickle.load(f)

    gts_arr = np.array([row['cat'] for row in extra['gts']])
    single_arr = extra['ensemble_single_scores']['cat']
    first_arr = extra['first_max_scores']['cat']
    second_arr = extra['second_max_scores']['cat']
    fusion_arr = 0.5 * extra['ensemble_single_scores']['cat'] + 0.5 * extra['second_max_scores']['cat']

    print(np.unique(gts_arr))
    assert(np.all((gts_arr == 0) | (gts_arr == 1)))

    plt.clf()
    _, axs = plt.subplots(4, 1, figsize=(6.4, 4*0.5*(2/3)*4.8), sharex=True, sharey=True)
    axs[0].hist(single_arr[gts_arr > 0], density=True, bins=30, histtype='step', edgecolor='b', linewidth=1.5, label='gt pos')
    axs[0].hist(single_arr[gts_arr == 0], density=True, bins=30, histtype='step', edgecolor='r', linewidth=1.5, label='gt neg')
    axs[0].set_xlabel('singleton score', fontsize=15)
    axs[0].set_ylabel('density', fontsize=15)
    axs[0].set_title('"cat" class: singleton score histograms', fontsize=15)
    axs[0].legend(fontsize=14)
    axs[1].hist(first_arr[gts_arr > 0], density=True, bins=30, histtype='step', edgecolor='b', linewidth=1.5, label='gt pos')
    axs[1].hist(first_arr[gts_arr == 0], density=True, bins=30, histtype='step', edgecolor='r', linewidth=1.5, label='gt neg')
    axs[1].set_xlabel('1max score', fontsize=15)
    axs[1].set_ylabel('density', fontsize=15)
    axs[1].set_title('"cat" class: 1max score histograms', fontsize=15)
    axs[1].legend(fontsize=14)
    axs[2].hist(second_arr[gts_arr > 0], density=True, bins=30, histtype='step', edgecolor='b', linewidth=1.5, label='gt pos')
    axs[2].hist(second_arr[gts_arr == 0], density=True, bins=30, histtype='step', edgecolor='r', linewidth=1.5, label='gt neg')
    axs[2].set_xlabel('2max score', fontsize=15)
    axs[2].set_ylabel('density', fontsize=15)
    axs[2].set_title('"cat" class: 2max score histograms', fontsize=15)
    axs[2].legend(fontsize=14)
    axs[3].hist(fusion_arr[gts_arr > 0], density=True, bins=30, histtype='step', edgecolor='b', linewidth=1.5, label='gt pos')
    axs[3].hist(fusion_arr[gts_arr == 0], density=True, bins=30, histtype='step', edgecolor='r', linewidth=1.5, label='gt neg')
    axs[3].set_xlabel('0.5 * singleton + 0.5 * 2max', fontsize=15)
    axs[3].set_ylabel('density', fontsize=15)
    axs[3].set_title('"cat" class: fuse 2max with singleton', fontsize=15)
    axs[3].legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.clf()


if __name__ == '__main__':
    make_qualitative_histograms(*(sys.argv[1:]))
