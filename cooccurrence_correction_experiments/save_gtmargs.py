import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, get_data_manager


GTMARGS_FILENAME_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/%s_gtmargs.pkl'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}


def save_gtmargs(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    gtmargs = np.mean(gts, axis=0)
    with open(GTMARGS_FILENAME_DICT[dataset_name], 'wb') as f:
        pickle.dump({'classnames' : classnames, 'mat' : gtmargs}, f)


def usage():
    print('Usage: python save_gtmargs.py <dataset_name>')


if __name__ == '__main__':
    save_gtmargs(*(sys.argv[1:]))
