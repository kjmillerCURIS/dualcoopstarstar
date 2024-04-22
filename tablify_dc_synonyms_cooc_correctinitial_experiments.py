import os
import sys
sys.path.append('cooccurrence_correction_experiments')
import glob
import pickle
from tqdm import tqdm
from compute_mAP import compute_mAP_main


OUTPUT_BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output')
TABLE_FILENAME = os.path.join(OUTPUT_BASE_DIR, '../plots_and_tables/dc_synonyms_cooc_correctinitial_3.27.2024.csv')
BETTER_POLYSEMY_LIST = [1]
APPEND_CLASS_LIST = [1]
ORIG_CLASSNAMES_INIT_LIST = [1]
THREE_SEP_LIST = [0]
MAT_NAME_LIST = ['gt_epsilon0.25_zeta0.25','expectedsurprise_confonly','expectedsurprise_fillin_20loc_3lpc', 'expectedsurprise_fillin_missingmiddle_20loc_3lpchigh_1lpclow']
LOSS_TYPE_LIST = ['prob_stopgrad_logit', 'correlation']
ALPHA_LIST = {'prob_stopgrad_logit' : [0.25], 'correlation' : [0.0075]}
BETA_OVER_ALPHA_LIST = {'prob_stopgrad_logit' : [0.5], 'correlation' : [0.5]}


def get_baseline_results_dir():
    return os.path.join(OUTPUT_BASE_DIR, 'frozen_pseudolabel_with_skipping_coco_seed1/Caption_tri_wta_soft_pseudolabel/rn101/nctx21_cscTrue_ctpend/seed1/results')


def get_results_dir(better_polysemy, append_class, orig_classnames_init, three_sep, mat_name, loss_type, alpha, beta_over_alpha):
    llm_name = 'synonyms8' + ('_betterPolysemy' if better_polysemy else '') + ('_appendOrigClassname' if append_class else '')
    job_id = 'frozen_pseudolabel_wskip_%s_originit%d_threesep%d_ternary_cooccurrence_correctinitial_%s_%s_%.5f_%.5f_coco_seed1'%(llm_name, orig_classnames_init, three_sep, loss_type, mat_name, alpha, beta_over_alpha)
    return os.path.join(OUTPUT_BASE_DIR, job_id, 'Caption_tri_wta_soft_pseudolabel_learnedensemble_ternary_cooccurrence_correctinitial/rn101/nctx21_cscTrue_ctpend/seed1/results')


#return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, corrected_pseudo_mAP
#will omit corrected_pseudo_mAP if omit_corrected_pseudo_mAP=True
def process_results_dir(results_dir, omit_corrected_pseudo_mAP=False):
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

    init_pseudo_mAP = compute_mAP_main(os.path.join(results_dir, '../pseudolabels.pth.tar-init'))
    if omit_corrected_pseudo_mAP:
        return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP
    else:
        corrected_pseudo_mAP = compute_mAP_main(os.path.join(results_dir, '../pseudolabels.pth.tar-init_corrected'))
        return best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, corrected_pseudo_mAP


#def tablify_dc_cooc_correctinitial_experiments():
#    f = open(TABLE_FILENAME, 'w')
#    f.write('method,diagonal bug?,cooccurrence source,cooccurrence loss,alpha,beta_over_alpha,initial pseudolabel mAP,corrected pseudolabel mAP,best_val_mAP,best_epoch,last_saved_epoch\n')
#    baseline_results_dir = get_baseline_results_dir()
#    baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch, baseline_init_pseudo_mAP = process_results_dir(baseline_results_dir, omit_corrected_pseudo_mAP=True)
#    f.write('baseline,no,N/A,N/A,N/A,N/A,%f,NA,%f,%d,%d\n'%(baseline_init_pseudo_mAP, baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch))
#    for mat_name in tqdm(MAT_NAME_LIST):
#        for loss_type in LOSS_TYPE_LIST:
#            for alpha in ALPHA_LIST[loss_type]:
#                for beta_over_alpha in BETA_OVER_ALPHA_LIST[loss_type]:
#                    for diagonal_bug in [False, True]:
#                        results_dir = get_results_dir(diagonal_bug, mat_name, loss_type, alpha, beta_over_alpha)
#                        if not os.path.exists(results_dir):
#                            print('MISSING results for: (%s, %s, %s, %f, %f)'%(str(diagonal_bug), mat_name, loss_type, alpha, beta_over_alpha))
#                            continue
#
#                        best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, corrected_pseudo_mAP = process_results_dir(results_dir)
#                        f.write('correct initial pseudolabels,%s,%s,%s,%f,%f,%f,%f,%f,%d,%d\n'%({'True' : 'yes', 'False' : 'no'}[str(diagonal_bug)], mat_name, loss_type, alpha, beta_over_alpha, init_pseudo_mAP, corrected_pseudo_mAP, best_val_mAP, best_epoch, last_saved_epoch))
#
#    f.close()


def yesno(onezero):
    return ('yes' if onezero else 'no')


def tablify_dc_synonyms_cooc_correctinitial_experiments():
    f = open(TABLE_FILENAME, 'w')
    f.write('method,synonym strategy,append orig classname?,pseudolabel init strategy,three separate ensembles?,cooccurrence source,cooccurrence loss,alpha,beta_over_alpha,initial pseudolabel mAP,corrected pseudolabel mAP,best_val_mAP,best_epoch,last_saved_epoch\n')
    baseline_results_dir = get_baseline_results_dir()
    baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch, baseline_init_pseudo_mAP = process_results_dir(baseline_results_dir, omit_corrected_pseudo_mAP=True)
    f.write('baseline,N/A,N/A,use orig classnames only,N/A,N/A,N/A,N/A,N/A,%f,N/A,%f,%d,%d\n'%(baseline_init_pseudo_mAP, baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch))
    for better_polysemy in BETTER_POLYSEMY_LIST:
        for append_class in APPEND_CLASS_LIST:
            for orig_classnames_init in ORIG_CLASSNAMES_INIT_LIST:
                for three_sep in THREE_SEP_LIST:
                    for mat_name in MAT_NAME_LIST:
                        for loss_type in LOSS_TYPE_LIST:
                            for alpha in ALPHA_LIST[loss_type]:
                                for beta_over_alpha in BETA_OVER_ALPHA_LIST[loss_type]:
                                    results_dir = get_results_dir(better_polysemy, append_class, orig_classnames_init, three_sep, mat_name, loss_type, alpha, beta_over_alpha)
                                    if not os.path.exists(results_dir):
                                        print('MISSING results for: (%d, %d, %d, %d, %s, %s, %f, %f)'%(better_polysemy, append_class, orig_classnames_init, three_sep, mat_name, loss_type, alpha, beta_over_alpha))
                                        assert(False)
                                        continue

                                    best_val_mAP,best_epoch,last_saved_epoch,init_pseudo_mAP,corrected_pseudo_mAP = process_results_dir(results_dir)
                                    synonym_strategy = {0 : 'simple strategy', 1 : 'polysemy strategy'}[better_polysemy]
                                    pseudolabel_init_strategy = {0 : 'use synonyms', 1 : 'use orig classnames only'}[orig_classnames_init]
                                    f.write('synonyms+cooc(correctinitial),%s,%s,%s,%s,%s,%s,%f,%f,%f,%f,%f,%d,%d\n'%(synonym_strategy, yesno(append_class), pseudolabel_init_strategy, yesno(three_sep), mat_name, loss_type, alpha, beta_over_alpha, init_pseudo_mAP, corrected_pseudo_mAP, best_val_mAP, best_epoch, last_saved_epoch))

    f.close()


if __name__ == '__main__':
    tablify_dc_synonyms_cooc_correctinitial_experiments()
