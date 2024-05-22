import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from compute_mAP import average_precision
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT
from compute_joint_marg_disagreement import compute_joint_marg_disagreement


CONFLICT_THRESHOLD_LIST = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2]
P_LIST = [0.01, 0.02, 0.03, 0.05, 0.1, 0.25]
Q_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
R_LIST = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
MAP_YLIM_DICT = {'COCO2014_partial' : (0,100), 'nuswide_partial' : (0,100), 'VOC2007_partial' : (0,100)}
DATA_PERCENTAGE_YLIM = (0,100)
P_COLOR_LIST = ['r', 'lawngreen', 'b', 'k', 'orange', 'gray']
Q_COLOR_LIST = ['r', 'lawngreen', 'b', 'k', 'orange', 'gray']
ABLATION_FIGSIZE = (5*6.4, 4.8)
TREATMENT_FIGSIZE = (5*6.4, 6*4.8)


def compute_pseudolabel_mask_confidentonly(scores, p, q):
    pos_thresholds = np.quantile(scores, 1-p, axis=0, keepdims=True)
    neg_thresholds = np.quantile(scores, q, axis=0, keepdims=True)
    return ((scores > pos_thresholds) | (scores < neg_thresholds)).astype('int32')


#scores is 2D numpy array
#joint_over_margs should be Pr(i,j) / (Pr(i) * Pr(j)), plain and simple
#p is for confident positives
#q is for confident negatives
#r is for killer positives
#returns mask which has 1 if included and 0 if excluded
def compute_pseudolabel_mask(scores, joint_over_margs, conflict_threshold, p, q, r):
    confident_mask = compute_pseudolabel_mask_confidentonly(scores, p, q)
    conflict_matrix = (joint_over_margs < conflict_threshold).astype('int32')
    np.fill_diagonal(conflict_matrix, 0)
    killer_thresholds = np.quantile(scores, 1-r, axis=0, keepdims=True)
    killer_mask = (scores > killer_thresholds).astype('int32')
    killed_mask = killer_mask @ conflict_matrix
    mask = ((killed_mask == 0) | (confident_mask > 0)).astype('int32')
    return mask


def compute_mAP_with_mask(scores, gts, mask):
    APs = []
    for i in range(gts.shape[1]):
        one_scores = scores[mask[:,i] > 0,i]
        one_gts = gts[mask[:,i] > 0,i]
        one_AP = average_precision(one_scores, one_gts)
        APs.append(one_AP)

    return 100.0 * np.mean(APs)


def compute_data_percentages(gts, mask):
    pos_data_percentages = 100.0 * np.sum(gts * mask, axis=0) / np.sum(gts, axis=0)
    neg_data_percentages = 100.0 * np.sum((1 - gts) * mask, axis=0) / np.sum(1 - gts, axis=0)
    return pos_data_percentages, neg_data_percentages


#return stats_dict which has:
#-"mAP"
#-"mean_pos_data_percentage"
#-"mean_neg_data_percentage"
#-"worst_pos_data_percentage"
#-"worst_neg_data_percentage"
def process_one_setting(scores, gts, joint_over_margs, conflict_threshold, p, q, r, confidentonly=False):
    assert(np.all((gts == 0) | (gts == 1)))
    if confidentonly:
        assert(all([z is None for z in [joint_over_margs, conflict_threshold, r]]))
    else:
        assert(all([z is not None for z in [joint_over_margs, conflict_threshold, r]]))

    stats_dict = {}
    if confidentonly:
        mask = compute_pseudolabel_mask_confidentonly(scores, p, q)
    else:
        mask = compute_pseudolabel_mask(scores, joint_over_margs, conflict_threshold, p, q, r)

    stats_dict['mAP'] = compute_mAP_with_mask(scores, gts, mask)
    pos_data_percentages, neg_data_percentages = compute_data_percentages(gts, mask)
    stats_dict['mean_pos_data_percentage'] = np.mean(pos_data_percentages)
    stats_dict['mean_neg_data_percentage'] = np.mean(neg_data_percentages)
    stats_dict['worst_pos_data_percentage'] = np.amin(pos_data_percentages)
    stats_dict['worst_neg_data_percentage'] = np.amin(neg_data_percentages)
    return stats_dict


def load_scores_and_gts(dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    disagreement_mat = compute_joint_marg_disagreement(gts, dataset_name)
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        scores = pickle.load(f)

    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    scores = np.array([scores[impath] for impath in sorted(scores.keys())])
    return scores, gts


def get_plot_filename(dataset_name, q, confidentonly=False):
    plot_dir = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/pseudolabel_removal_plots')
    os.makedirs(plot_dir, exist_ok=True)
    if confidentonly:
        return os.path.join(plot_dir, '%s_pseudolabel_removal_confidentonly.png'%(dataset_name.split('_')[0]))
    else:
        return os.path.join(plot_dir, '%s_pseudolabel_removal_q%f.png'%(dataset_name.split('_')[0], q))


def plot_pseudolabel_removal_stats_ablations(dataset_name):
    scores, gts = load_scores_and_gts(dataset_name)
    plot_filename = get_plot_filename(dataset_name, None, confidentonly=True)
    plt.clf()
    _, axs = plt.subplots(1, 5, figsize=ABLATION_FIGSIZE, dpi=300)
    for q, color in tqdm(zip(Q_LIST, Q_COLOR_LIST)):
        mAP_list = []
        mean_pos_data_percentage_list = []
        mean_neg_data_percentage_list = []
        worst_pos_data_percentage_list = []
        worst_neg_data_percentage_list = []
        for p in P_LIST:
            stats_dict = process_one_setting(scores, gts, None, None, p, q, None, confidentonly=True)
            mAP_list.append(stats_dict['mAP'])
            mean_pos_data_percentage_list.append(stats_dict['mean_pos_data_percentage'])
            mean_neg_data_percentage_list.append(stats_dict['mean_neg_data_percentage'])
            worst_pos_data_percentage_list.append(stats_dict['worst_pos_data_percentage'])
            worst_neg_data_percentage_list.append(stats_dict['worst_neg_data_percentage'])

        for ax, my_list in zip(axs, [mAP_list, mean_pos_data_percentage_list, mean_neg_data_percentage_list, worst_pos_data_percentage_list, worst_neg_data_percentage_list]):
            ax.plot(P_LIST, my_list, color=color, label='q = %f'%(q))

    for ax, metric_type in zip(axs, ['mAP', 'mean_pos_data_percentage', 'mean_neg_data_percentage', 'worst_pos_data_percentage', 'worst_neg_data_percentage']):
        ax.legend()
        ax.set_xlabel('p')
        ax.set_ylabel(metric_type)
        ax.set_title('%s: confident thresholding only - %s'%(dataset_name.split('_')[0], metric_type))
        if metric_type == 'mAP':
            ax.set_ylim(MAP_YLIM_DICT[dataset_name])
            ax.set_ylim(DATA_PERCENTAGE_YLIM)

    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.clf()
    plt.close()


def plot_pseudolabel_removal_stats_treatment(dataset_name):
    scores, gts = load_scores_and_gts(dataset_name)
    joint_over_margs = 1 - compute_joint_marg_disagreement({i : gts[i,:] for i in range(gts.shape[0])}, dataset_name)
    for q in tqdm(Q_LIST):
        plot_filename = get_plot_filename(dataset_name, q)
        plt.clf()
        _, axs = plt.subplots(len(CONFLICT_THRESHOLD_LIST), 5, figsize=TREATMENT_FIGSIZE, dpi=300)
        for conflict_threshold, ax_row in zip(CONFLICT_THRESHOLD_LIST, axs):
            for p, color in zip(P_LIST, P_COLOR_LIST):
                mAP_list = []
                mean_pos_data_percentage_list = []
                mean_neg_data_percentage_list = []
                worst_pos_data_percentage_list = []
                worst_neg_data_percentage_list = []
                for r in R_LIST:
                    stats_dict = process_one_setting(scores, gts, joint_over_margs, conflict_threshold, p, q, r)
                    mAP_list.append(stats_dict['mAP'])
                    mean_pos_data_percentage_list.append(stats_dict['mean_pos_data_percentage'])
                    mean_neg_data_percentage_list.append(stats_dict['mean_neg_data_percentage'])
                    worst_pos_data_percentage_list.append(stats_dict['worst_pos_data_percentage'])
                    worst_neg_data_percentage_list.append(stats_dict['worst_neg_data_percentage'])

                for ax, my_list in zip(ax_row, [mAP_list, mean_pos_data_percentage_list, mean_neg_data_percentage_list, worst_pos_data_percentage_list, worst_neg_data_percentage_list]):
                    ax.plot(R_LIST, my_list, color=color, label='(q, conflict_threshold, p) = (%f, %f, %f)'%(q, conflict_threshold, p))

                for ax, metric_type in zip(ax_row, ['mAP', 'mean_pos_data_percentage', 'mean_neg_data_percentage', 'worst_pos_data_percentage', 'worst_neg_data_percentage']):
                    ax.legend()
                    ax.set_xlabel('r')
                    ax.set_ylabel(metric_type)
                    ax.set_title('%s: pseudolabel-removal, (q, conflict_threhold) = (%f, %f) - %s'%(dataset_name.split('_')[0], q, conflict_threshold, metric_type))
                    if metric_type == 'mAP':
                        ax.set_ylim(MAP_YLIM_DICT[dataset_name])

                    ax.set_ylim(DATA_PERCENTAGE_YLIM)

        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.clf()
        plt.close()


def plot_pseudolabel_removal_stats(dataset_name):
    plot_pseudolabel_removal_stats_ablations(dataset_name)
    plot_pseudolabel_removal_stats_treatment(dataset_name)


def usage():
    print('Usage: python plot_pseudolabel_removal_stats.py <dataset_name>')


if __name__ == '__main__':
    plot_pseudolabel_removal_stats(*(sys.argv[1:]))
