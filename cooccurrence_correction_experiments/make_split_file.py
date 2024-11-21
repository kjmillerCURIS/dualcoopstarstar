import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
from harvest_training_gts import TESTING_GTS_FILENAME_DICT


def make_split_file(dataset_name):
    testing_gts_filename = TESTING_GTS_FILENAME_DICT[dataset_name]
    with open(testing_gts_filename, 'rb') as f:
        gts = pickle.load(f)

    f = open(os.path.join(os.path.dirname(testing_gts_filename), '%s_test_split.txt'%(dataset_name.split('_')[0])), 'w')
    for impath in tqdm(sorted(gts.keys())):
        indices = np.nonzero(np.array(gts[impath]) == 1)[0]
        f.write(os.path.splitext(os.path.basename(impath))[0] + ' ' + ' '.join([str(i) for i in indices]) + '\n')

    f.close()


def usage():
    print('Usage: python make_split_file.py <dataset_name>')


if __name__ == '__main__':
    make_split_file(*(sys.argv[1:]))
