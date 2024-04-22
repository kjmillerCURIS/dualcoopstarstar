import os
import sys
import pickle
from harvest_training_gts import get_data_manager


NAMER_DICT = {'COCO2014_partial' : 'mscoco', 'nuswide_partial' : 'nuswide', 'VOC2007_partial' : 'voc2007'}


def save_classnames(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    out_filename = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/%s_classnames.pkl'%(NAMER_DICT[dataset_name]))
    with open(out_filename, 'wb') as f:
        pickle.dump(classnames, f)


def usage():
    print('Usage: python save_classnames.py <dataset_name>')


if __name__ == '__main__':
    save_classnames(*(sys.argv[1:]))
