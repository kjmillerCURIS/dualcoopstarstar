import os
import sys
import copy
import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from tqdm import tqdm
from harvest_training_gts import get_data_manager, NUM_CLASSES_DICT, TRAINING_GTS_FILENAME_DICT
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT


C_FILENAME_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training_gts-disagreement_C.pkl'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/nuswide_training_gts-disagreement_C.pkl'), 'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/voc2007_training_gts-disagreement_C.pkl')}
CLUSTER_SORT_FILENAME_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training_gts-disagreement_cluster_sort.pkl'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/nuswide_training_gts-disagreement_cluster_sort.pkl'), 'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/voc2007_training_gts-disagreement_cluster_sort.pkl')}
EPSILON_LIST = [0.25, 0.5, 0.75]
ZETA_LIST = [-0.25, -0.5, -0.75]
VIS_MIN = -1
VIS_MIN_BIG = -40
VIS_MAX = 1
CLUSTER_SORT_FLOOR = -2
PLOT_PREFIX_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/disagreement_plots/mscoco_training_gts-disagreement'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/disagreement_plots/nuswide_training_gts-disagreement'), 'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/disagreement_plots/voc2007_training_gts-disagreement')}
EPSILON_TO_USE = 0.25
ZETA_TO_USE = -0.25
TERNARY_COOCCURRENCE_MAT_FILENAME_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/ternary_cooccurrence_mats/COCO2014_partial/gt_epsilon0.25_zeta0.25.pkl'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/ternary_cooccurrence_mats/nuswide_partial/gt_epsilon0.25_zeta0.25.pkl'), 'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/ternary_cooccurrence_mats/VOC2007_partial/gt_epsilon0.25_zeta0.25.pkl')}


#instance_probs[impath] = [0,1]^N
#returns NxN heatmap of 1 - Pr(i,j) / (Pr(i) * Pr(j))
#positive for things that do NOT co-occur, negative for things that DO co-occur
#instance_probs could be binary or continuous, as long as they're between 0 and 1
def compute_joint_marg_disagreement(instance_probs, dataset_name):
    marg_probs = np.zeros(NUM_CLASSES_DICT[dataset_name])
    joint_probs = np.zeros((NUM_CLASSES_DICT[dataset_name], NUM_CLASSES_DICT[dataset_name]))
    for impath in tqdm(sorted(instance_probs.keys())):
        p = instance_probs[impath]
        assert(np.amin(p) >= 0.0)
        assert(np.amax(p) <= 1.0)
        marg_probs += p
        joint_probs += p[:,np.newaxis] @ p[np.newaxis,:]

    marg_probs /= len(instance_probs)
    joint_probs /= len(instance_probs)
    indep_probs = marg_probs[:,np.newaxis] @ marg_probs[np.newaxis,:]
    disagreement = 1 - joint_probs / indep_probs
    return disagreement


def plot_heatmap(my_heatmap, classnames, plot_suffix, dataset_name, cluster_sort=None, vis_min=None, vis_max=None, is_mask=False, my_title=None, plot_filename=None):
    assert((plot_suffix is None) != (plot_filename is None)) #this is an XOR
    my_heatmap = copy.deepcopy(my_heatmap) #so things are non-destructive
    classnames = copy.deepcopy(classnames)
    if cluster_sort is not None:
        if isinstance(cluster_sort, bool):
            assert(not is_mask)
            Y = copy.deepcopy(my_heatmap)
            Y = np.maximum(Y, CLUSTER_SORT_FLOOR)
            np.fill_diagonal(Y, np.amin(Y))
            Y -= np.amin(Y)
            Y = ssd.squareform(Y)
            Z = sch.linkage(Y, method='average')
            cluster_sort = sch.leaves_list(sch.optimal_leaf_ordering(Z, Y))

        my_heatmap = my_heatmap[cluster_sort][:, cluster_sort]
        classnames = [classnames[cs] for cs in cluster_sort]

    np.fill_diagonal(my_heatmap, 0) #otherwise we'd have perfect correlation on diagonal

    plt.clf()
    plt.figure(figsize=(12, 10))
    hmap = plt.imshow(my_heatmap, vmin=vis_min, vmax=vis_max, aspect='auto')
    plt.xticks(ticks=np.arange(len(classnames)), labels=classnames, rotation=90)
    plt.yticks(ticks=np.arange(len(classnames)), labels=classnames)
    cbar = plt.colorbar(hmap)
    tick_max = None
    if not is_mask:
        tick_min = np.amin(my_heatmap)
        tick_max = np.amax(my_heatmap)
        if vis_min is not None:
            tick_min = vis_min

        if vis_max is not None:
            tick_max = vis_max

        cbar.set_ticks([tick_min, 0, tick_max])
        cbar.set_ticklabels([f'{tick_min:.2f}', '0', f'{tick_max:.2f}'])

    if not os.path.exists(os.path.dirname(PLOT_PREFIX_DICT[dataset_name])):
        os.makedirs(os.path.dirname(PLOT_PREFIX_DICT[dataset_name]))

    if my_title is not None:
        plt.title(my_title)

    plt.tight_layout()
    if plot_suffix is not None:
        plt.savefig(PLOT_PREFIX_DICT[dataset_name] + '-' + plot_suffix, dpi=300)
    else:
        plt.savefig(plot_filename, dpi=300)

    plt.clf()
    return cluster_sort


def save_ternary_cooccurrence_mat(disagreement, classnames, dataset_name):
    mat = np.zeros_like(disagreement)
    mat[disagreement > EPSILON_TO_USE] = 1
    mat[disagreement < ZETA_TO_USE] = -1
    mat = np.triu(mat, k=1)
    ternary_cooccurrence_mat = {'mat' : mat, 'classnames' : classnames}
    with open(TERNARY_COOCCURRENCE_MAT_FILENAME_DICT[dataset_name], 'wb') as f:
        pickle.dump(ternary_cooccurrence_mat, f)


def compute_joint_marg_disagreement_main(dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gt_labels = pickle.load(f)

    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    disagreement = compute_joint_marg_disagreement(gt_labels, dataset_name)
    
    with open(C_FILENAME_DICT[dataset_name], 'wb') as f:
        pickle.dump(disagreement, f)

    save_ternary_cooccurrence_mat(disagreement, classnames, dataset_name)
    cluster_sort = plot_heatmap(disagreement, classnames, 'heatmap-clustersort-vmin.png', dataset_name, cluster_sort=True, vis_min=VIS_MIN, vis_max=VIS_MAX)
    with open(CLUSTER_SORT_FILENAME_DICT[dataset_name], 'wb') as f:
        pickle.dump(cluster_sort, f)

    plot_heatmap(disagreement, classnames, 'heatmap-clustersort.png', dataset_name, vis_min=VIS_MIN_BIG, vis_max=VIS_MAX, cluster_sort=cluster_sort)

    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    pseudolabel_probs = {impath : 1 / (1 + np.exp(-pseudolabel_logits[impath])) for impath in sorted(pseudolabel_logits.keys())}
    pseudo_disagreement = compute_joint_marg_disagreement(pseudolabel_probs, dataset_name)

    plot_heatmap(pseudo_disagreement, classnames, 'pseudo-heatmap-clustersort-vmin.png', dataset_name, cluster_sort=cluster_sort, vis_min=VIS_MIN, vis_max=VIS_MAX)
    plot_heatmap(pseudo_disagreement, classnames, 'pseudo-heatmap-clustersort.png', dataset_name, cluster_sort=cluster_sort, vis_min=VIS_MIN_BIG, vis_max=VIS_MAX)

    for epsilon in EPSILON_LIST:
        plot_heatmap((disagreement > epsilon), classnames, 'heatmap-clustersort-mask-epsilon%f.png'%(epsilon), dataset_name, cluster_sort=cluster_sort, is_mask=True)
    
    for zeta in ZETA_LIST:
        plot_heatmap((disagreement > zeta), classnames, 'heatmap-clustersort-mask-zeta%f.png'%(zeta), dataset_name, cluster_sort=cluster_sort, is_mask=True)


def usage():
    print('Usage: python compute_joint_marg_disagreement.py <dataset_name>')


if __name__ == '__main__':
    compute_joint_marg_disagreement_main(*(sys.argv[1:]))
