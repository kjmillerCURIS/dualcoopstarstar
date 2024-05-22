import os
import sys
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT


NUM_SAMPLES = 300


def plot_pseudolabel_sample(dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    gt_probs = np.array([gts[impath] for impath in sorted(gts.keys())])
    pseudo_probs = np.array([1 / (1 + np.exp(-pseudolabel_logits[impath])) for impath in sorted(pseudolabel_logits.keys())])
    random.seed(0)
    indices = random.sample(range(gt_probs.shape[0]), NUM_SAMPLES)
    gt_probs = gt_probs[indices, :]
    pseudo_probs = pseudo_probs[indices, :]
    plt.clf()
    plt.figure(figsize=(15, 50))
    plt.imshow(gt_probs, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('%s: gts sample'%(dataset_name.split('_')[0]))
    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_gts_sample.png'%(dataset_name.split('_')[0]), dpi=300)
    plt.clf()
    plt.figure(figsize=(15, 50))
    plt.imshow(pseudo_probs, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('%s: CLIP pseudolabels sample'%(dataset_name.split('_')[0]))
    plt.savefig('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_CLIP_pseudolabels_sample.png'%(dataset_name.split('_')[0]), dpi=300)
    plt.clf()
    numIa = cv2.imread('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_gts_sample.png'%(dataset_name.split('_')[0]))
    numIb = cv2.imread('../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_CLIP_pseudolabels_sample.png'%(dataset_name.split('_')[0]))
    numIc = np.hstack([numIa, numIb])
    assert(numIc is not None)
    print('me')
    outfilename = '../vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/%s_samples.png'%(dataset_name.split('_')[0])
    assert(os.path.exists(os.path.dirname(outfilename)))
    cv2.imwrite(outfilename, numIc)
    print('ow')


def usage():
    print('Usage: python plot_pseudolabel_sample.py <dataset_name>')


if __name__ == '__main__':
    plot_pseudolabel_sample(*(sys.argv[1:]))
