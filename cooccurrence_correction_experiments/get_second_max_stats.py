import os
import sys
import glob
import numpy as np
import pickle
from tqdm import tqdm


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
MODEL_TYPES = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14', 'ViT-L14336px']


def compute_second_max_stats_helper(dataset_name, baseline_key, treatment_key):
    diffs = []
    for model_type in MODEL_TYPES:
        result_filename = os.path.join(BASE_DIR, '%s_test/result_files/%s_test_%s_results.pkl'%(dataset_name.split('_')[0], dataset_name.split('_')[0], model_type))
        with open(result_filename, 'rb') as f:
            result = pickle.load(f)

        diffs.append(result[treatment_key]['mAP'] - result[baseline_key]['mAP'])

    avg_diff = np.mean(diffs)
    min_diff = np.amin(diffs)
    max_diff = np.amax(diffs)
    print('%s mAP - %s mAP: avg=%f (min=%f, max=%f)'%(treatment_key, baseline_key, avg_diff, min_diff, max_diff))


def compute_second_max_stats(dataset_name):
    compute_second_max_stats_helper(dataset_name, 'ensemble_single_uncalibrated', 'ensemble_single_calibrated')
    compute_second_max_stats_helper(dataset_name, 'ensemble_single_calibrated', 'third_max_avg')
    compute_second_max_stats_helper(dataset_name, 'third_max_avg', 'third_max_pca')
    compute_second_max_stats_helper(dataset_name, 'ensemble_single_calibrated', 'third_max_pca')


def usage():
    print('Usage: python compute_second_max_stats.py <dataset_name>')


if __name__ == '__main__':
    compute_second_max_stats(*(sys.argv[1:]))
