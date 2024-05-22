import os
import sys
from tqdm import tqdm
from tablify_dc_conflict_cleaninitial_experiments import OUTPUT_BASE_DIR, get_baseline_results_dir, get_zeroshot_results_dir, get_results_dir, process_results_dir


TABLE_FILENAME_DICT = {dataset_name : os.path.join(OUTPUT_BASE_DIR, '../plots_and_tables/%s_dc_conflict_cleaninitial_uegtmargs.csv'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
CONFLICT_THRESHOLD_LIST = [0.001, 0.2]
PQ_LIST = [0.0, 0.1, 0.5, 0.9]
R_LIST = [0.1, 0.5, 0.9]


def tablify_dc_conflict_cleaninitial_usegtmargs_experiments(dataset_name):
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
        for pq in PQ_LIST:
            for r in R_LIST:
                results_dir = get_results_dir(conflict_threshold, True, pq, pq, r, dataset_name)
                if not os.path.exists(results_dir):
                    print('MISSING results for: (%f, %f, %f, %f)'%(conflict_threshold, pq, pq, r))
                    continue

                best_val_mAP, best_epoch, last_saved_epoch, init_pseudo_mAP, cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage = process_results_dir(results_dir, dataset_name, ordered_impaths_cache)
                f.write('conflict clean (YES gtmargs),%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n'%(conflict_threshold, pq, pq, r, init_pseudo_mAP, cleaned_pseudo_mAP, mean_pos_data_percentage, worst_pos_data_percentage, mean_neg_data_percentage, worst_neg_data_percentage, best_val_mAP, best_epoch, last_saved_epoch))

    f.close()


def usage():
    print('Usage: python tablify_dc_conflict_cleaninitial_usegtmargs_experiments.py <dataset_name>')


if __name__ == '__main__':
    tablify_dc_conflict_cleaninitial_usegtmargs_experiments(*(sys.argv[1:]))
