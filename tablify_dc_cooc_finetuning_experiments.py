import os
import sys
import glob
import pickle
from tqdm import tqdm


#only tablify gt_cooc
#start with 1 row for baseline
#outermost is lod (0, 1, 2, 3)
#then alpha (small to large)
#then beta_over_alpha (small to large)
#should report method ("baseline" vs "finetune dc with gt_cooc"), lod, alpha, beta_over_alpha, best_val_mAP, best_epoch, last_saved_epoch


OUTPUT_BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output')
TABLE_FILENAME = os.path.join(OUTPUT_BASE_DIR, '../plots_and_tables/gt_cooc_dc_finetune_3.13.2024.csv')
LOD_LIST = [0, 1, 3]
ALPHA_LIST = [5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.0625, 0.125]
BETA_OVER_ALPHA_LIST = [0.0, 0.25, 0.5, 0.75]


def get_baseline_results_dir():
    return os.path.join(OUTPUT_BASE_DIR, 'frozen_pseudolabel_with_skipping_coco_seed1/Caption_tri_wta_soft_pseudolabel/rn101/nctx21_cscTrue_ctpend/seed1/results')


#will return None if it's missing
def get_results_dir(diagonal_bug, lod, alpha, beta_over_alpha):
    if diagonal_bug:
        if lod > 0:
            job_id = 'frozen_pseudolabel_wskip_ternary_cooccurrence_lod%d_prob_stopgrad_logit_gt_epsilon0.25_zeta0.25_%.5f_%.5f_coco_seed1'%(lod, alpha, beta_over_alpha)
        else:
            assert(lod == 0)
            job_id = 'frozen_pseudolabel_wskip_ternary_cooccurrence_prob_stopgrad_logit_gt_epsilon0.25_zeta0.25_%.5f_%.5f_coco_seed1'%(alpha, beta_over_alpha)

        return os.path.join(OUTPUT_BASE_DIR, 'TRIU_BUG', job_id, 'Caption_tri_wta_soft_pseudolabel_ternary_cooccurrence/rn101/nctx21_cscTrue_ctpend/seed1/results')

    assert(not diagonal_bug)
    job_id = 'frozen_pseudolabel_wskip_ternary_cooccurrence_lod%d_prob_stopgrad_logit_gt_epsilon0.25_zeta0.25_%.5f_%.5f_coco_seed1'%(lod, alpha, beta_over_alpha)
    return os.path.join(OUTPUT_BASE_DIR, job_id, 'Caption_tri_wta_soft_pseudolabel_ternary_cooccurrence/rn101/nctx21_cscTrue_ctpend/seed1/results')


#return best_val_mAP, best_epoch, last_saved_epoch
def process_results_dir(results_dir):
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

    return best_val_mAP, best_epoch, last_saved_epoch


def tablify_dc_cooc_finetuning_experiments():
    f = open(TABLE_FILENAME, 'w')
    f.write('method,diagonal bug?,delay cooc loss for #epochs,alpha,beta_over_alpha,best_val_mAP,best_epoch,last_saved_epoch\n')
    baseline_results_dir = get_baseline_results_dir()
    baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch = process_results_dir(baseline_results_dir)
    f.write('baseline,no,N/A,N/A,N/A,%f,%d,%d\n'%(baseline_best_val_mAP, baseline_best_epoch, baseline_last_saved_epoch))
    for lod in tqdm(LOD_LIST):
        for alpha in ALPHA_LIST:
            for beta_over_alpha in BETA_OVER_ALPHA_LIST:
                for diagonal_bug in [False, True]:
                    results_dir = get_results_dir(diagonal_bug, lod, alpha, beta_over_alpha)
                    if not os.path.exists(results_dir):
                        print('MISSING results for: (%s, %d, %f, %f)'%(str(diagonal_bug), lod, alpha, beta_over_alpha))
                        continue

                    best_val_mAP, best_epoch, last_saved_epoch = process_results_dir(results_dir)
                    f.write('finetune dc with gt cooc loss,%s,%d,%f,%f,%f,%d,%d\n'%({'True' : 'yes', 'False' : 'no'}[str(diagonal_bug)], lod, alpha, beta_over_alpha, best_val_mAP, best_epoch, last_saved_epoch))

    f.close()


if __name__ == '__main__':
    tablify_dc_cooc_finetuning_experiments()
