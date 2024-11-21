import os
import sys
import random
import string
from tqdm import tqdm
from harvest_training_gts import get_data_manager
from second_max_experiments import COCO_KEVIN_CLASSNAMES_RENAMER


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
NUM_WAFFLES = 30
WAFFLE_LEN = 10
RANDOM_SEED = 0


def make_waffle_prompts(dataset_name):
    random.seed(RANDOM_SEED)
    waffles = []
    for _ in tqdm(range(NUM_WAFFLES)):
        waffles.append(''.join(random.choices(string.ascii_letters + string.digits, k=WAFFLE_LEN)))

    print(waffles)

    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    if dataset_name == 'COCO2014_partial':
        classnames = [(COCO_KEVIN_CLASSNAMES_RENAMER[c] if c in COCO_KEVIN_CLASSNAMES_RENAMER else c) for c in classnames]

    prompts = []
    for classname in tqdm(classnames):
        prompts.append('A photo of a ' + classname + '.\n')

    for waffle in tqdm(waffles):
        for classname in classnames:
            prompts.append('A photo of a ' + classname + ' which is ' + waffle + '.\n')

    output_filename = os.path.join(BASE_DIR, dataset_name.split('_')[0] + '_test', '%s_waffle_prompts.txt'%(dataset_name.split('_')[0]))
    f = open(output_filename, 'w')
    for prompt in tqdm(prompts):
        f.write(prompt)

    f.close()


if __name__ == '__main__':
    make_waffle_prompts(*(sys.argv[1:]))
