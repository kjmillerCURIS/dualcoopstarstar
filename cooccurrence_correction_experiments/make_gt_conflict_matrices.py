import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, get_data_manager
from compute_joint_marg_disagreement import compute_joint_marg_disagreement


#compute_joint_marg_disagreement(instance_probs, dataset_name)
CONFLICT_THRESHOLD_LIST = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2]


def make_gt_conflict_matrices(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    disagreements = compute_joint_marg_disagreement(gts, dataset_name)
    conflicts = 1 - disagreements
    for conflict_threshold in CONFLICT_THRESHOLD_LIST:
        conflict_matrix = np.triu((conflicts < conflict_threshold).astype('float64'), k=1)
        conflict_matrix_filename = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/conflict_matrices/%s/gt_conflict_matrix_threshold%f.pkl'%(dataset_name, conflict_threshold))
        os.makedirs(os.path.dirname(conflict_matrix_filename), exist_ok=True)
        with open(conflict_matrix_filename, 'wb') as f:
            pickle.dump({'mat' : conflict_matrix, 'classnames' : classnames}, f)


def usage():
    print('Usage: python make_gt_conflict_matrices.py <dataset_name>')


if __name__ == '__main__':
    make_gt_conflict_matrices(*(sys.argv[1:]))
