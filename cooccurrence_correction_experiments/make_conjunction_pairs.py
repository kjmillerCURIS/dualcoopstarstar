import os
import sys
import copy
from tqdm import tqdm


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
NUM_CLASSES = {'COCO2014_partial' : 80, 'VOC2007_partial' : 20, 'nuswideTheirVersion_partial' : 81}
CONJUNCTION_MODE_DICT = {'or' : 'CONJUNCTION_OR', 'with' : 'CONJUNCTION_WITH', 'next to' : 'CONJUNCTION_NEXT_TO', 'not' : 'CONJUNCTION_NOT', 'all_conjunctions' : 'ALL_CONJUNCTIONS'}


#return list of prompts
def process_one_prompt(prompt, conjunction_mode):
    assert(conjunction_mode in ['or', 'with', 'next to', 'not', 'all_conjunctions'])
    assert(',' not in prompt)
    assert(len(prompt.split(' and ')) == 2)
    A, B = prompt.split(' and ')
    if conjunction_mode in ['or', 'with', 'next to']:
        return ['%s %s %s'%(A, conjunction_mode, B)]
    elif conjunction_mode == 'not':
        return [A + ' and ' + B, A + ' and not ' + B, B + ' and not ' + A]
    else:
        assert(conjunction_mode == 'all_conjunctions')
        retval = []
        for cm in ['or', 'with', 'next to', 'not']:
            retval.extend(process_one_prompt(prompt, cm))

        return retval


def make_conjunction_pairs_one_dataset_one_mode(dataset_name, conjunction_mode):
    f = open(os.path.join(BASE_DIR, '%s_test'%(dataset_name.split('_')[0]), 'PAIR_ONLY', '%s_simple_single_and_compound_prompts.txt'%(dataset_name.split('_')[0])), 'r')
    classnames = []
    pair_prompts = []
    for line in f:
        if len(classnames) < NUM_CLASSES[dataset_name]:
            classnames.append(line.rstrip('\n'))
        else:
            pair_prompts.append(line.rstrip('\n'))

    f.close()
    print((classnames[0], classnames[-1], pair_prompts[0], pair_prompts[-1]))

    new_pair_prompts = []
    for p in pair_prompts:
        new_pair_prompts.extend(process_one_prompt(p, conjunction_mode))

    os.makedirs(os.path.join(BASE_DIR, '%s_test'%(dataset_name.split('_')[0]), CONJUNCTION_MODE_DICT[conjunction_mode]), exist_ok=True)
    f = open(os.path.join(BASE_DIR, '%s_test'%(dataset_name.split('_')[0]), CONJUNCTION_MODE_DICT[conjunction_mode], '%s_simple_single_and_compound_prompts.txt'%(dataset_name.split('_')[0])), 'w')
    f.write('\n'.join(classnames + new_pair_prompts))
    f.close()


def make_conjunction_pairs_one_dataset(dataset_name):
    for conjunction_mode in ['or', 'with', 'next to', 'not', 'all_conjunctions']:
        make_conjunction_pairs_one_dataset_one_mode(dataset_name, conjunction_mode)


def make_conjunction_pairs():
    for dataset_name in tqdm(['COCO2014_partial', 'VOC2007_partial', 'nuswideTheirVersion_partial']):
        make_conjunction_pairs_one_dataset(dataset_name)


if __name__ == '__main__':
    make_conjunction_pairs()
