import os
import sys
import copy
from tqdm import tqdm


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
NUM_CLASSES = {'COCO2014_partial' : 80, 'VOC2007_partial' : 20, 'nuswideTheirVersion_partial' : 81}


def make_all_pairs_one_dataset(dataset_name):
    f = open(os.path.join(BASE_DIR, '%s_test'%(dataset_name.split('_')[0]), '%s_simple_single_and_compound_prompts.txt'%(dataset_name.split('_')[0])), 'r')
    classnames = []
    for line in f:
        classnames.append(line.rstrip('\n'))
        if len(classnames) >= NUM_CLASSES[dataset_name]:
            break

    f.close()
    assert(len(classnames) == NUM_CLASSES[dataset_name])
    things = copy.deepcopy(classnames)
    for i in range(len(classnames) - 1):
        for j in range(i + 1, len(classnames)):
            a, b = classnames[i], classnames[j]
            if dataset_name == 'COCO2014_partial':
                a = a.replace('food bowl', 'bowl').capitalize()
                b = b.replace('food bowl', 'bowl').capitalize()

            things.append(a + ' and ' + b)

    f = open(os.path.join(BASE_DIR, '%s_test'%(dataset_name.split('_')[0]), 'ALL_PAIRS', '%s_simple_single_and_compound_prompts.txt'%(dataset_name.split('_')[0])), 'w')
    f.write('\n'.join(things))
    f.close()


def make_all_pairs():
    for dataset_name in tqdm(['COCO2014_partial', 'VOC2007_partial', 'nuswideTheirVersion_partial']):
        make_all_pairs_one_dataset(dataset_name)


if __name__ == '__main__':
    make_all_pairs()
