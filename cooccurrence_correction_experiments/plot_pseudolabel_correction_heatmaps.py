import os
import sys
import cv2
import glob
import numpy as np
import pickle
from matplotlib import pyplot as plt
import torch
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, get_data_manager
from compute_joint_marg_disagreement import compute_joint_marg_disagreement, plot_heatmap
from compute_mAP import get_pseudolabels_as_dict


#def get_pseudolabels_as_dict(checkpoint, gt_labels, dataset_name):
#def compute_joint_marg_disagreement(instance_probs, dataset_name):


JOB_DIR_BASE = '../vislang-domain-exploration-data/dualcoopstarstar-data/output'
JOB_DIR_PROB_STOPGRAD_LOGIT_DICT = {'COCO2014_partial' : os.path.join(JOB_DIR_BASE, 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_gt_epsilon0.25_zeta0.25_0.25000_0.50000_coco_seed1'), 'nuswide_partial' : os.path.join(JOB_DIR_BASE, 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_prob_stopgrad_logit_gt_epsilon0.25_zeta0.25_0.25000_0.50000_nuswide_seed1'), 'VOC2007_partial' : os.path.join(JOB_DIR_BASE, 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_prob_stopgrad_logit_gt_epsilon0.25_zeta0.25_0.25000_0.50000_voc2007_seed1')}
JOB_DIR_CORRELATION_DICT = {'COCO2014_partial' : os.path.join(JOB_DIR_BASE, 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_correlation_gt_epsilon0.25_zeta0.25_0.00750_0.50000_coco_seed1'), 'nuswide_partial' : os.path.join(JOB_DIR_BASE, 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_correlation_gt_epsilon0.25_zeta0.25_0.00750_0.50000_nuswide_seed1'), 'VOC2007_partial' : os.path.join(JOB_DIR_BASE, 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_correlation_gt_epsilon0.25_zeta0.25_0.00750_0.50000_voc2007_seed1')}
PLOT_FILENAME_GT_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/correction_stats_plots/%s_disagreement_gt_labels.png'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
PLOT_FILENAME_INIT_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/correction_stats_plots/%s_disagreement_clip_pseudolabels.png'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
PLOT_FILENAME_PSL_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/correction_stats_plots/%s_disagreement_prob_stopgrad_logit_corrected_pseudolabels.png'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
PLOT_FILENAME_CORRELATION_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/correction_stats_plots/%s_disagreement_correlation_corrected_pseudolabels.png'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
PLOT_FILENAME_STITCH_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/correction_stats_plots/%s_disagreement_stitch.png'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}


#returns gt_disagreement, initial_disagreement, corrected_disagreement
#these are 2D arrays of 1 - Pr(i,j) / ((Pr(i) * Pr(j))
def obtain_disagreements(job_dir, dataset_name):
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
    return gt_disagreement, initial_disagreement, corrected_disagreement


def make_stitch(dataset_name):
    numIa = cv2.imread(PLOT_FILENAME_GT_DICT[dataset_name])
    numIb = cv2.imread(PLOT_FILENAME_INIT_DICT[dataset_name])
    numIc = cv2.imread(PLOT_FILENAME_PSL_DICT[dataset_name])
    numId = cv2.imread(PLOT_FILENAME_CORRELATION_DICT[dataset_name])
    numIstitch = np.vstack([np.hstack([numIa, numIb]), np.hstack([numIc, numId])])
    cv2.imwrite(PLOT_FILENAME_STITCH_DICT[dataset_name], numIstitch)


def plot_pseudolabel_correction_heatmaps(dataset_name):
    job_dir_psl = JOB_DIR_PROB_STOPGRAD_LOGIT_DICT[dataset_name]
    job_dir_correlation = JOB_DIR_CORRELATION_DICT[dataset_name]
    gt_disagreement, initial_disagreement, psl_disagreement = obtain_disagreements(job_dir_psl, dataset_name)
    _, _, correlation_disagreement = obtain_disagreements(job_dir_correlation, dataset_name)
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    cluster_sort = plot_heatmap(gt_disagreement, classnames, None, dataset_name, cluster_sort=True, vis_min=-1, vis_max=1, my_title='%s: gt labels'%(dataset_name.split('_')[0]), plot_filename=PLOT_FILENAME_GT_DICT[dataset_name])
    plot_heatmap(initial_disagreement, classnames, None, dataset_name, cluster_sort=cluster_sort, vis_min=-1, vis_max=1, my_title='%s: CLIP pseudolabels'%(dataset_name.split('_')[0]), plot_filename=PLOT_FILENAME_INIT_DICT[dataset_name])
    plot_heatmap(psl_disagreement, classnames, None, dataset_name, cluster_sort=cluster_sort, vis_min=-1, vis_max=1, my_title='%s: prob_stopgrad_logits corrected pseudolabels'%(dataset_name.split('_')[0]), plot_filename=PLOT_FILENAME_PSL_DICT[dataset_name])
    plot_heatmap(correlation_disagreement, classnames, None, dataset_name, cluster_sort=cluster_sort, vis_min=-1, vis_max=1, my_title='%s: correlation corrected pseudolabels'%(dataset_name.split('_')[0]), plot_filename=PLOT_FILENAME_CORRELATION_DICT[dataset_name])
    make_stitch(dataset_name)


def usage():
    print('Usage: python plot_pseudolabel_correction_heatmaps.py <dataset_name>')


if __name__ == '__main__':
    plot_pseudolabel_correction_heatmaps(*(sys.argv[1:]))
