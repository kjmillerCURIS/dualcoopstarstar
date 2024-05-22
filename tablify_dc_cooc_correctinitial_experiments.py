import os
import sys
sys.path.append('cooccurrence_correction_experiments')
import glob
import pickle
from tqdm import tqdm
from compute_mAP import compute_mAP_main


OUTPUT_BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output')
TABLE_FILENAME_DICT = {dataset_name : os.path.join(OUTPUT_BASE_DIR, '../plots_and_tables/%s_dc_cooc_correctinitial.csv'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
MAT_NAME_LIST = ['gt_epsilon0.25_zeta0.25','expectedsurprise_confonly','expectedsurprise_fillin_20loc_3lpc', 'expectedsurprise_fillin_missingmiddle_20loc_3lpchigh_1lpclow', 'expectedsurprise_removepositives_20loc_1lpc']
LOSS_TYPE_LIST = ['prob_stopgrad_logit', 'correlation']
ALPHA_LIST = {'prob_stopgrad_logit' : [0.25], 'correlation' : [0.0075]}
BETA_OVER_ALPHA_LIST = {'prob_stopgrad_logit' : [0.5], 'correlation' : [0.5]}
JOBPART_DICT = {'COCO2014_partial' : 'coco', 'nuswide_partial' : 'nuswide', 'VOC2007_partial' : 'voc2007'}
RN_DICT = {'COCO2014_partial' : 'rn101', 'nuswide_partial' : 'rn101_nus', 'VOC2007_partial' : 'rn101_bn96'}


def get_baseline_results_dir(dataset_name):
    return os.path.join(OUTPUT_BASE_DIR, 'frozen_pseudolabel_with_skipping_%s_seed1/Caption_tri_wta_soft_pseudolabel/%s/nctx21_cscTrue_ctpend/seed1/results'%(JOBPART_DICT[dataset_name], RN_DICT[dataset_name]))


def get_zeroshot_results_dir(use_cossim, dataset_name):
    return os.path.join(OUTPUT_BASE_DIR, 'zsclip_%sensemble80_%s/Caption_tri_wta_soft_pseudolabel/%s/nctx21_cscTrue_ctpend/seed1/results'%({True : 'use_cossim_', False : ''}[use_cossim], JOBPART_DICT[dataset_name], RN_DICT[dataset_name]))


#will return None if it's missing
def get_results_dir(diagonal_bug, mat_name, loss_type, alpha, beta_over_alpha, dataset_name):
    if loss_type == 'prob_stopgrad_logit' and dataset_name == 'COCO2014_partial':
        loss_type_str = ''
    else:
        loss_type_str = loss_type + '_'

    job_id = 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_%s%s_%.5f_%.5f_%s_seed1'%(loss_type_str, mat_name, alpha, beta_over_alpha, JOBPART_DICT[dataset_name])
    if diagonal_bug:
        assert(dataset_name == 'COCO2014_partial')
        return os.path.join(OUTPUT_BASE_DIR, 'TRIU_BUG', job_id, 'Caption_tri_wta_soft_pseudolabel_ternary_cooccurrence_correctinitial/%s/nctx21_cscTrue_ctpend/seed1/results'%(RN_DICT[dataset_name]))
    else:
        return os.path.join(OUTPUT_BASE_DIR, job_id, 'Caption_tri_wta_soft_pseudolabel_ternary_cooccurrence_correctinitial/%s/nctx21_cscTrue_ctpend/seed1/results'%(RN_DICT[dataset_name]))


#return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, corrected_pseudo_mAP
#will omit init_pseudo_mAP, corrected_pseudo_mAP if omit_pseudo_mAP=True
#will omit corrected_pseudo_mAP if omit_corrected_pseudo_mAP=True
def process_results_dir(results_dir, dataset_name, omit_pseudo_mAP=False, omit_corrected_pseudo_mAP=False):
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

    init_pseudo_mAP = compute_mAP_main(os.path.join(results_dir, '../pseudolabels.pth.tar-init'), dataset_name)
    if omit_corrected_pseudo_mAP:
        return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP

    corrected_pseudo_mAP = compute_mAP_main(os.path.join(results_dir, '../pseudolabels.pth.tar-init_corrected'), dataset_name)
    return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, corrected_pseudo_mAP


def tablify_dc_cooc_correctinitial_experiments(dataset_name):
    f = open(TABLE_FILENAME_DICT[dataset_name], 'w')
    f.write('method,diagonal bug?,cooccurrence source,cooccurrence loss,alpha,beta_over_alpha,initial pseudolabel mAP,corrected pseudolabel mAP,best_val_mAP,best_epoch,last_saved_epoch\n')
    zs_cossim_results_dir = get_zeroshot_results_dir(True, dataset_name)
    zs_cossim_mAP, _, _ = process_results_dir(zs_cossim_results_dir, dataset_name, omit_pseudo_mAP=True)
    f.write('ZSCLIP raw cossim,no,N/A,N/A,N/A,N/A,N/A,N/A,%f,N/A,N/A\n'%(zs_cossim_mAP))
    zs_softmax_results_dir = get_zeroshot_results_dir(False, dataset_name)
    zs_softmax_mAP, _, _ = process_results_dir(zs_softmax_results_dir, dataset_name, omit_pseudo_mAP=True)
    f.write('ZSCLIP softmax,no,N/A,N/A,N/A,N/A,N/A,N/A,%f,N/A,N/A\n'%(zs_softmax_mAP))
    baseline_results_dir = get_baseline_results_dir(dataset_name)
    baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch, baseline_init_pseudo_mAP = process_results_dir(baseline_results_dir, dataset_name, omit_corrected_pseudo_mAP=True)
    f.write('baseline,no,N/A,N/A,N/A,N/A,%f,NA,%f,%d,%d\n'%(baseline_init_pseudo_mAP, baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch))
    for mat_name in tqdm(MAT_NAME_LIST):
        for loss_type in LOSS_TYPE_LIST:
            for alpha in ALPHA_LIST[loss_type]:
                for beta_over_alpha in BETA_OVER_ALPHA_LIST[loss_type]:
                    for diagonal_bug in [False, True]:
                        if diagonal_bug and (dataset_name != 'COCO2014_partial'):
                            continue

                        results_dir = get_results_dir(diagonal_bug, mat_name, loss_type, alpha, beta_over_alpha, dataset_name)
                        if not os.path.exists(results_dir):
                            print('MISSING results for: (%s, %s, %s, %f, %f)'%(str(diagonal_bug), mat_name, loss_type, alpha, beta_over_alpha))
                            continue

                        best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, corrected_pseudo_mAP = process_results_dir(results_dir, dataset_name)
                        f.write('correct initial pseudolabels,%s,%s,%s,%f,%f,%f,%f,%f,%d,%d\n'%({'True' : 'yes', 'False' : 'no'}[str(diagonal_bug)], mat_name, loss_type, alpha, beta_over_alpha, init_pseudo_mAP, corrected_pseudo_mAP, best_val_mAP, best_epoch, last_saved_epoch))

    f.close()


def usage():
    print('Usage: python tablify_dc_cooc_correctinitial_experiments.py <dataset_name>')


if __name__ == '__main__':
    tablify_dc_cooc_correctinitial_experiments(*(sys.argv[1:]))
