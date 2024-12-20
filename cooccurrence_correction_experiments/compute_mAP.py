import os
import sys
import numpy as np
import pickle
import torch
from tqdm import tqdm
from dassl.evaluation import build_evaluator
from harvest_training_gts import get_cfg, get_data_manager, TRAINING_GTS_FILENAME_DICT
sys.path.append('.')
import datasets.coco2014_partial
import datasets.nuswide_trainset_gt
import datasets.voc2007_partial
sys.path.pop()


#standalone function
def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def compute_mAP(logits, gt_labels, dataset_name):
    cfg = get_cfg(dataset_name)
    evaluator = build_evaluator(cfg)
    assert(evaluator.is_for_dualcoopstarstar)
    evaluator.reset()
    #just one "batch"
    impaths = sorted(gt_labels.keys())
    assert(sorted(logits.keys()) == impaths)
    logits_batch = torch.tensor([logits[impath] for impath in impaths])
    gt_labels_batch = np.array([gt_labels[impath] for impath in impaths])
    gt_labels_batch = 2 * gt_labels_batch - 1 #{0,1} ==> {-1,1}
    assert(np.all((gt_labels_batch == -1) | (gt_labels_batch == 1)))
    gt_labels_batch = torch.tensor(gt_labels_batch)
    evaluator.process({'default' : logits_batch}, gt_labels_batch)
    res = evaluator.evaluate()
    return res['mAP']


#returns logits
def get_pseudolabels_as_dict(checkpoint, gt_labels, dataset_name):
    logits_arr = checkpoint['pseudolabel_logits'].numpy()
    logits = {}
    dm = get_data_manager(dataset_name)
    for item, logits_vec in zip(dm.dataset.train_x, logits_arr):
        assert(item.impath not in logits)
        assert(np.all(np.maximum(item.label, 0) == gt_labels[item.impath]))
        logits[item.impath] = logits_vec

    assert(len(logits) == len(dm.dataset.train_x))
    assert(len(logits) == len(logits_arr))
    assert(len(logits) == len(gt_labels))
    return logits


def get_ordered_impaths(dataset_name):
    dm = get_data_manager(dataset_name)
    return [item.impath for item in dm.dataset.train_x]


def compute_mAP_main(pseudolabels_filename, dataset_name, return_epoch=False):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gt_labels = pickle.load(f)

    checkpoint = torch.load(pseudolabels_filename, map_location='cpu')
    logits = get_pseudolabels_as_dict(checkpoint, gt_labels, dataset_name)
    mAP = compute_mAP(logits, gt_labels, dataset_name)
    print(mAP)
    if return_epoch:
        return mAP, checkpoint['epoch']
    else:
        return mAP


def usage():
    print('Usage: python compute_mAP.py <pseudolabels_filename> <dataset_name>')


if __name__ == '__main__':
    compute_mAP_main(*(sys.argv[1:]))
