import os
import sys
import numpy as np
import torch
from tqdm import tqdm


def multiquantile(X, q):
    assert(len(X.shape) == 2)
    assert(q.shape == (X.shape[1],))
    thresholds = torch.stack([torch.quantile(X[:,i], q[i]) for i in range(X.shape[1])], dim=-1)
    assert(thresholds.shape == (X.shape[1],))
    thresholds = torch.unsqueeze(thresholds, 0)
    assert(thresholds.shape == (1, X.shape[1]))
    return thresholds


def clean_initial_pseudolabels_confidentonly(pseudolabel_logits, cfg, gtmargs=None):
    with torch.no_grad():
        if cfg.TRAIN.USE_GTMARGS:
            assert(gtmargs is not None)
            assert(gtmargs.shape == (pseudolabel_logits.shape[1],))
            pos_thresholds = multiquantile(pseudolabel_logits, 1-(cfg.TRAIN.CLEAN_P*gtmargs))
            neg_thresholds = multiquantile(pseudolabel_logits, cfg.TRAIN.CLEAN_Q*gtmargs)
        else:
            pos_thresholds = torch.quantile(pseudolabel_logits, 1 - cfg.TRAIN.CLEAN_P, dim=0, keepdim=True)
            neg_thresholds = torch.quantile(pseudolabel_logits, cfg.TRAIN.CLEAN_Q, dim=0, keepdim=True)

        pseudolabel_weights = ((pseudolabel_logits < neg_thresholds) | (pseudolabel_logits > pos_thresholds)).to(pseudolabel_logits.dtype)

    return pseudolabel_weights


#pseudolabel_weights should have 1's for confident negatives and confident positives
#we'll add ones in the middle that aren't eliminated by conflict
def do_conflict_part(pseudolabel_logits, pseudolabel_weights, conflict_matrix, cfg, gtmargs=None):
    with torch.no_grad():
        if cfg.TRAIN.USE_GTMARGS:
            assert(gtmargs is not None)
            assert(gtmargs.shape == (pseudolabel_logits.shape[1],))
            killer_thresholds = multiquantile(pseudolabel_logits,1-(cfg.TRAIN.CLEAN_R*gtmargs))
        else:
            killer_thresholds = torch.quantile(pseudolabel_logits, 1 - cfg.TRAIN.CLEAN_R, dim=0, keepdim=True)

        killer_mask = (pseudolabel_logits > killer_thresholds).to(pseudolabel_weights.dtype)
        killed_mask = (killer_mask @ conflict_matrix.to(pseudolabel_weights.dtype) > 0)
        pseudolabel_weights = ((~killed_mask) | (pseudolabel_weights > 0)).to(pseudolabel_weights.dtype)

    return pseudolabel_weights


def clean_initial_pseudolabels(pseudolabel_logits, conflict_matrix, cfg, gtmargs=None):
    pseudolabel_weights = clean_initial_pseudolabels_confidentonly(pseudolabel_logits, cfg, gtmargs=gtmargs)
    if cfg.TRAIN.CONFIDENTONLY:
        return pseudolabel_weights

    pseudolabel_weights = do_conflict_part(pseudolabel_logits, pseudolabel_weights, conflict_matrix, cfg, gtmargs=gtmargs)
    return pseudolabel_weights
