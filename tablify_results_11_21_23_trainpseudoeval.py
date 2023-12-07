import os
import sys
from matplotlib import pyplot as plt
from result_utils_11_21_23 import extract_mAPs_given_hparams_trainpseudoeval, extract_mAPs_given_runID, extract_zsclip_mAPs


TABLE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/plots_and_tables')


def get_best(ret):
    best_mAP = float('-inf')
    best_epoch = None
    for epoch, mAP in zip(ret['epoch'], ret['mAP']):
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch

    return best_mAP, best_epoch


def get_last(ret):
    last_mAP = None
    last_epoch = float('-inf')
    for epoch, mAP in zip(ret['epoch'], ret['mAP']):
        if epoch > last_epoch:
            last_epoch = epoch
            last_mAP = mAP

    return last_mAP, last_epoch


def noyes(x):
    return {0:'no',1:'yes'}[x]


def tablify_results_11_21_23_trainpseudoeval():
    rows = []
    rows.append(['algo_type','adjust_logits','use_bias','bandwidth','stepsize','last_mAP','last_epoch','best_mAP','best_epoch'])
    for do_adjust_logits in [1]: #[0,1]
        for use_bias in [0,1]:
            for bandwidth in [0.1, 0.2, 0.4]:
                for stepsize in [0.0625, 0.125, 0.25, 0.5, 1.]:
                    ret = extract_mAPs_given_hparams_trainpseudoeval(do_adjust_logits,use_bias,bandwidth,stepsize)
                    best_mAP, best_epoch = get_best(ret)
                    last_mAP, last_epoch = get_last(ret)
                    row = ['DualCoOp++ w/CDUL',noyes(do_adjust_logits),noyes(use_bias),'%.2f'%(bandwidth),'%.5f'%(stepsize),'%.1f'%(last_mAP),str(last_epoch),'%.1f'%(best_mAP),str(best_epoch)]
                    rows.append(row)

    f = open(os.path.join(TABLE_DIR, '11_21_23-trainpseudolabel-table.csv'), 'w')
    for row in rows:
        f.write(','.join(row) + '\n')

    f.close()


if __name__ == '__main__':
    tablify_results_11_21_23_trainpseudoeval()
