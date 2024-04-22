import os
import sys
import glob
import numpy as np
import pickle
from matplotlib import pyplot as plt
import torch
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT
from compute_joint_marg_disagreement import compute_joint_marg_disagreement
from compute_mAP import get_pseudolabels_as_dict


#def get_pseudolabels_as_dict(checkpoint, gt_labels, dataset_name):
#def compute_joint_marg_disagreement(instance_probs, dataset_name):


DISAGREEMENT_FLOOR = 1e-5
BUFFER_RATIO = 0.025
VIS_THRESHOLDS = [0.2, 0.1, 0.05, 0.01, 0.001]
PLOT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/correction_stats_plots')


def triu_flatten(A):
    assert(len(A.shape) == 2)
    assert(A.shape[0] == A.shape[1])
    row_indices, col_indices = np.triu_indices(A.shape[0], k=1)
    return A[row_indices, col_indices]


#returns gt_stats, initial_stats, corrected_stats
#these are all lists/vectors
#"stat" refers to log(Pr(i,j) / (Pr(i) * Pr(j)))
def obtain_stats(job_dir, dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gt_labels = pickle.load(f)

    initial_pseudolabels_filenames = sorted(glob.glob(os.path.join(job_dir, '*/*/*/*', 'pseudolabels.pth.tar-init')))
    assert(len(initial_pseudolabels_filenames) == 1)
    initial_pseudolabels_filename = initial_pseudolabels_filenames[0]
    corrected_pseudolabels_filenames = sorted(glob.glob(os.path.join(job_dir, '*/*/*/*', 'pseudolabels.pth.tar-init_corrected')))
    assert(len(corrected_pseudolabels_filenames) == 1)
    corrected_pseudolabels_filename = corrected_pseudolabels_filenames[0]
    initial_pseudolabels_checkpoint = torch.load(initial_pseudolabels_filename, map_location='cpu')
    corrected_pseudolabels_checkpoint = torch.load(corrected_pseudolabels_filename, map_location='cpu')
    initial_pseudolabels_logits = get_pseudolabels_as_dict(initial_pseudolabels_checkpoint, gt_labels, dataset_name)
    corrected_pseudolabels_logits = get_pseudolabels_as_dict(corrected_pseudolabels_checkpoint, gt_labels, dataset_name)
    initial_pseudolabels_probs = {impath : 1 / (1 + np.exp(-initial_pseudolabels_logits[impath])) for impath in sorted(initial_pseudolabels_logits.keys())}
    corrected_pseudolabels_probs = {impath : 1 / (1 + np.exp(-corrected_pseudolabels_logits[impath])) for impath in sorted(corrected_pseudolabels_logits.keys())}
    gt_disagreement = compute_joint_marg_disagreement(gt_labels, dataset_name)
    initial_disagreement = compute_joint_marg_disagreement(initial_pseudolabels_probs, dataset_name)
    corrected_disagreement = compute_joint_marg_disagreement(corrected_pseudolabels_probs, dataset_name)
    disagreement_floor = DISAGREEMENT_FLOOR
    gt_stats = triu_flatten(np.log(np.maximum(1 - gt_disagreement, disagreement_floor)))
    initial_stats = triu_flatten(np.log(np.maximum(1 - initial_disagreement, disagreement_floor)))
    corrected_stats = triu_flatten(np.log(np.maximum(1 - corrected_disagreement, disagreement_floor)))
    return gt_stats, initial_stats, corrected_stats


def get_plot_filename(job_dir):
    os.makedirs(PLOT_DIR, exist_ok=True)
    return os.path.join(PLOT_DIR, 'correction_stats-%s.png'%(os.path.basename(job_dir)))


def make_plot(job_dir, gt_stats, initial_stats, corrected_stats):
    plot_filename = get_plot_filename(job_dir)
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    axs[0].scatter(gt_stats, initial_stats, color='b', marker='.')
    axs[1].scatter(gt_stats, corrected_stats, color='b', marker='.')
    all_data = np.concatenate([gt_stats, initial_stats, corrected_stats])
    my_lim = (np.min(all_data), np.max(all_data))
    buffer = BUFFER_RATIO * (my_lim[1] - my_lim[0])
    my_lim = (my_lim[0] - buffer, my_lim[1] + buffer)
    for ax in axs:
        ax.plot(my_lim, my_lim, linestyle='solid', color='k')
        ax.vlines([np.log(vt) for vt in VIS_THRESHOLDS], my_lim[0], my_lim[1], linestyle='dashed', color='orange')
        ax.hlines([np.log(vt) for vt in VIS_THRESHOLDS], my_lim[0], my_lim[1], linestyle='dashed', color='orange')
        for vt in VIS_THRESHOLDS:
            ax.text(np.log(vt), 0.1 * my_lim[0] + 0.9 * my_lim[1], 'log(%.3f)'%(vt))
            ax.text(0.1 * my_lim[0] + 0.9 * my_lim[1], np.log(vt), 'log(%.3f)'%(vt))

        ax.set_xlim(my_lim)
        ax.set_ylim(my_lim)
        ax.set_xlabel('log(Pr_gt(i,j) / (Pr_gt(i) * Pr_gt(j)))')

    axs[0].set_ylabel('log(Pr_init(i,j) / (Pr_init(i) * Pr_init(j)))')
    axs[1].set_ylabel('log(Pr_corrected(i,j) / (Pr_corrected(i) * Pr_corrected(j)))')
    axs[0].set_title('initial pseudolabel cooccurrence stats vs gt')
    axs[1].set_title('corrected pseudolabel cooccurrence stats vs gt')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.clf()


def plot_pseudolabel_correction_stats(job_dir, dataset_name):
    gt_stats, initial_stats, corrected_stats = obtain_stats(job_dir, dataset_name)
    make_plot(job_dir, gt_stats, initial_stats, corrected_stats)


def usage():
    print('Usage: python plot_pseudolabel_correction_stats.py <job_dir> <dataset_name>')


if __name__ == '__main__':
    plot_pseudolabel_correction_stats(*(sys.argv[1:]))
