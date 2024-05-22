import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, get_data_manager
from compute_joint_marg_disagreement import compute_joint_marg_disagreement


def get_plot_filename(i, j, classnames, dataset_name):
    dataset_nickname = dataset_name.split('_')[0]
    classnickname_i = classnames[i].replace(' ', '')
    classnickname_j = classnames[j].replace(' ', '')
    return os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/pair_plots/%s/%s_%s-vs-%s_pair_plot.png'%(dataset_nickname, dataset_nickname, classnickname_i, classnickname_j))


def plot_pairs_one(i, j, scores, gts, disagreement_mat, classnames, dataset_name):
    plot_filename = get_plot_filename(i, j, classnames, dataset_name)
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.clf()
    points_00 = ((gts[:,i] == 0) & (gts[:,j] == 0))
    points_01 = ((gts[:,i] == 0) & (gts[:,j] == 1))
    points_10 = ((gts[:,i] == 1) & (gts[:,j] == 0))
    points_11 = ((gts[:,i] == 1) & (gts[:,j] == 1))
    plt.figure(figsize=(3*6.4, 3*4.8), dpi=600)
    plt.scatter(scores[points_00, i], scores[points_00, j], color='k', marker='.', s=1, label='"%s"=0, "%s"=0'%(classnames[i], classnames[j]))
    plt.scatter(scores[points_01, i], scores[points_01, j], color='r', marker='.', s=1, label='"%s"=0, "%s"=1'%(classnames[i], classnames[j]))
    plt.scatter(scores[points_10, i], scores[points_10, j], color='b', marker='.', s=1, label='"%s"=1, "%s"=0'%(classnames[i], classnames[j]))
    plt.scatter(scores[points_11, i], scores[points_11, j], color='lawngreen', marker='.', s=1, label='"%s"=1, "%s"=1'%(classnames[i], classnames[j]))
    plt.legend()
    plt.xlabel('"%s" logit'%(classnames[i]))
    plt.ylabel('"%s" logit'%(classnames[j]))
    plt.title('"%s" vs "%s" logits, 1 - Pr(i,j)/(Pr(i)*Pr(j)) = %f'%(classnames[i], classnames[j], disagreement_mat[i,j]))
    plt.savefig(plot_filename)
    plt.clf()
    plt.close()


def plot_pairs(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    disagreement_mat = compute_joint_marg_disagreement(gts, dataset_name)
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        scores = pickle.load(f)

    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    scores = np.array([scores[impath] for impath in sorted(scores.keys())])
    ij_list = []
    for i in range(0, gts.shape[1] - 1):
        for j in range(i + 1, gts.shape[1]):
            ij_list.append((i,j))

    for ij in tqdm(ij_list):
        i,j = ij
        plot_pairs_one(i, j, scores, gts, disagreement_mat, classnames, dataset_name)


def usage():
    print('Usage: python plot_pairs.py <dataset_name>')


if __name__ == '__main__':
    plot_pairs(*(sys.argv[1:]))
