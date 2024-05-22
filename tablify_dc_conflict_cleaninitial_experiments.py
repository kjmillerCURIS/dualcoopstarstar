import os
import sys
sys.path.append('cooccurrence_correction_experiments')
import glob
import numpy as np
import pickle
import torch
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT
from compute_mAP import get_ordered_impaths
from plot_pseudolabel_removal_stats import compute_mAP_with_mask


OUTPUT_BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output')
TABLE_FILENAME_DICT = {dataset_name : os.path.join(OUTPUT_BASE_DIR, '../plots_and_tables/%s_dc_conflict_cleaninitial.csv'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
CONFLICT_THRESHOLD_LIST = [0.001, 0.2]
P_LIST = [0.01, 0.1]
Q_LIST = [0.1, 0.4]
R_LIST = [0.1, 0.02]
JOBPART_DICT = {'COCO2014_partial' : 'coco', 'nuswide_partial' : 'nuswide', 'VOC2007_partial' : 'voc2007'}
RN_DICT = {'COCO2014_partial' : 'rn101', 'nuswide_partial' : 'rn101_nus', 'VOC2007_partial' : 'rn101_bn96'}


def get_baseline_results_dir(dataset_name):
    return os.path.join(OUTPUT_BASE_DIR, 'frozen_pseudolabel_with_skipping_%s_seed1/Caption_tri_wta_soft_pseudolabel/%s/nctx21_cscTrue_ctpend/seed1/results'%(JOBPART_DICT[dataset_name], RN_DICT[dataset_name]))


def get_zeroshot_results_dir(use_cossim, dataset_name):
    return os.path.join(OUTPUT_BASE_DIR, 'zsclip_%sensemble80_%s/Caption_tri_wta_soft_pseudolabel/%s/nctx21_cscTrue_ctpend/seed1/results'%({True : 'use_cossim_', False : ''}[use_cossim], JOBPART_DICT[dataset_name], RN_DICT[dataset_name]))


#will return None if it's missing
def get_results_dir(conflict_threshold, usegtmargs, p, q, r, dataset_name):
    job_id = 'frozen_pseudolabel_wskip_conflict_cleaninitial_gt_conflict_matrix_threshold%f%s_p%.5f_q%.5f_r%.5f_%s_seed1'%(conflict_threshold, {True : '_usegtmargs', False : ''}[usegtmargs], p, q, r, JOBPART_DICT[dataset_name])
    return os.path.join(OUTPUT_BASE_DIR, job_id, 'Caption_tri_wta_soft_pseudolabel_conflict_cleaninitial/%s/nctx21_cscTrue_ctpend/seed1/results'%(RN_DICT[dataset_name]))


#return cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage
def compute_masked_mAP_from_checkpoint(checkpoint_filename, dataset_name, ordered_impaths_cache, assert_all_present=False):
    checkpoint = torch.load(checkpoint_filename, map_location='cpu')
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    if ordered_impaths_cache['ordered_impaths'] is None:
        ordered_impaths = get_ordered_impaths(dataset_name)
        ordered_impaths_cache['ordered_impaths'] = ordered_impaths
    else:
        ordered_impaths = ordered_impaths_cache['ordered_impaths']

    gts = np.array([gts[impath] for impath in ordered_impaths])
    scores = checkpoint['pseudolabel_logits'].numpy()
    mask = checkpoint['pseudolabel_weights'].numpy()
    assert(np.all((mask == 1) | (mask == 0)))
    if assert_all_present:
        assert(np.all(mask == 1))

    cleaned_pseudo_mAP = compute_mAP_with_mask(scores, gts, mask)
    pos_data_percentages = 100.0 * np.sum(gts * mask, axis=0) / np.sum(gts, axis=0)
    neg_data_percentages = 100.0 * np.sum((1 - gts) * mask, axis=0) / np.sum(1 - gts, axis=0)
    mean_pos_data_percentage = np.mean(pos_data_percentages)
    worst_pos_data_percentage = np.amin(pos_data_percentages)
    mean_neg_data_percentage = np.mean(neg_data_percentages)
    worst_neg_data_percentage = np.amin(neg_data_percentages)
    return cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage


#return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage
#will omit init_pseudo_mAP and everything after if omit_pseudo_mAP=True
#will omit cleaned_pseudo_mAP and everything after if omit_cleaned_pseudo_mAP=True
def process_results_dir(results_dir, dataset_name, ordered_impaths_cache, omit_pseudo_mAP=False, omit_cleaned_pseudo_mAP=False):
    results_filenames = sorted(glob.glob(os.path.join(results_dir, 'results-0*.pkl')))
    assert(all(['after_train' not in os.path.basename(s) for s in results_filenames]))
    best_val_mAP = float('-inf')
    best_epoch = None
    last_saved_epoch = float('-inf')
    for results_filename in results_filenames:
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)

        mAP = results['mAP']
        epoch = int(os.path.splitext(os.path.basename(results_filename))[0].split('-')[1])
        if mAP > best_val_mAP:
            best_val_mAP = mAP
            best_epoch = epoch

        if epoch > last_saved_epoch:
            last_saved_epoch = epoch

    if best_epoch is None:
        import pdb
        pdb.set_trace()

    if omit_pseudo_mAP:
        return best_val_mAP, best_epoch, last_saved_epoch

    init_pseudo_mAP, _, _, _, _ = compute_masked_mAP_from_checkpoint(os.path.join(results_dir, '../pseudolabels.pth.tar-init'), dataset_name, ordered_impaths_cache, assert_all_present=True)
    if omit_cleaned_pseudo_mAP:
        return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP

    cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage = compute_masked_mAP_from_checkpoint(os.path.join(results_dir, '../pseudolabels.pth.tar-init_cleaned'), dataset_name, ordered_impaths_cache)
    return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage


def tablify_dc_conflict_cleaninitial_experiments(dataset_name):
    ordered_impaths_cache = {'ordered_impaths' : None}
    f = open(TABLE_FILENAME_DICT[dataset_name], 'w')
    f.write('method,conflict_threshold,p,q,r,initial pseudolabel mAP,cleaned pseudolabel mAP,mean_pos_data_percentage,worst_pos_data_percentage,mean_neg_data_percentage,worst_neg_data_percentage,best_val_mAP,best_epoch,last_saved_epoch\n')
    zs_cossim_results_dir = get_zeroshot_results_dir(True, dataset_name)
    zs_cossim_mAP, _, _ = process_results_dir(zs_cossim_results_dir, dataset_name, ordered_impaths_cache, omit_pseudo_mAP=True)
    f.write('ZSCLIP raw cossim,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,%f,N/A,N/A\n'%(zs_cossim_mAP))
    zs_softmax_results_dir = get_zeroshot_results_dir(False, dataset_name)
    zs_softmax_mAP, _, _ = process_results_dir(zs_softmax_results_dir, dataset_name, ordered_impaths_cache, omit_pseudo_mAP=True)
    f.write('ZSCLIP softmax,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,%f,N/A,N/A\n'%(zs_softmax_mAP))
    baseline_results_dir = get_baseline_results_dir(dataset_name)
    baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch, baseline_init_pseudo_mAP = process_results_dir(baseline_results_dir, dataset_name, ordered_impaths_cache, omit_cleaned_pseudo_mAP=True)
    f.write('baseline,N/A,N/A,N/A,N/A,%f,N/A,N/A,N/A,N/A,N/A,%f,%d,%d\n'%(baseline_init_pseudo_mAP, baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch))
    for conflict_threshold in CONFLICT_THRESHOLD_LIST:
        for p in P_LIST:
            for q in Q_LIST:
                for r in R_LIST:
                    results_dir = get_results_dir(conflict_threshold, False, p, q, r, dataset_name)
                    if not os.path.exists(results_dir):
                        print('MISSING results for: (%f, %f, %f, %f)'%(conflict_threshold, p, q, r))
                        continue

                    best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage = process_results_dir(results_dir, dataset_name, ordered_impaths_cache)
                    f.write('conflict clean (no gtmargs),%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n'%(conflict_threshold, p, q, r, init_pseudo_mAP, cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage, best_val_mAP, best_epoch, last_saved_epoch))

    f.close()


def usage():
    print('Usage: python tablify_dc_conflict_cleaninitial_experiments.py <dataset_name>')


if __name__ == '__main__':
    tablify_dc_conflict_cleaninitial_experiments(*(sys.argv[1:]))
