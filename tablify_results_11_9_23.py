import os
import sys
from matplotlib import pyplot as plt
from result_utils_11_9_23 import extract_mAPs_given_hparams, extract_mAPs_given_runID, extract_random_chance_mAPs

#def extract_mAPs_given_hparams(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable):


TABLE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/plots_and_tables')


def get_best(ret):
    best_mAP = float('-inf')
    best_epoch = None
    for epoch, mAP in zip(ret['epoch'], ret['mAP']):
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch

    return best_mAP, best_epoch


def tablify_results_11_9_23():
    for meow in ['hungarian', 'max']:
        rows = []
        rows.append(['algo_type','encoder_layers','decoder_layers','prompt_mode','use_spatial_embedding','learnable_temperature','best_mAP','best_epoch'])
        random_chance_mAP = extract_random_chance_mAPs()[meow]
        rows.append(['chance','NA','NA','NA','NA','NA','%.1f'%(random_chance_mAP),'NA'])
        baseline_row = ['DualCoOp++ (baseline)','NA','NA','NA','NA','NA']
        ret_baseline = extract_mAPs_given_runID('baseline_coco2014_partial_tricoop_wta_soft_448_CSC_p0_5-pos200-ctx21_norm', is_baseline=True, results_dir_suffix='Caption_tri_wta_soft/rn101/nctx21_cscTrue_ctpend/seed1/results')
        best_mAP, best_epoch = get_best(ret_baseline)
        baseline_row.append('%.1f'%(best_mAP))
        baseline_row.append('%d'%(best_epoch))
        rows.append(baseline_row)
        for num_encoder_layers, num_decoder_layers in [(0,1),(0,2),(2,2)]:
            for prompt_mode in ['pos_and_neg_learnable_prompt', 'pos_only_fixed_prompt']:
                for spatial in [0,1]:
                    if spatial == 0:
                        continue
                    for temperature_is_learnable in [0,1]:
                        row = ['our idea',str(num_encoder_layers),str(num_decoder_layers),prompt_mode,{0:'no',1:'yes'}[spatial],{0:'no',1:'yes'}[temperature_is_learnable]]
                        ret = extract_mAPs_given_hparams(prompt_mode,num_encoder_layers,num_decoder_layers,spatial,spatial,temperature_is_learnable)
                        ret = ret[meow]
                        best_mAP, best_epoch = get_best(ret)
                        row.append('%.1f'%(best_mAP))
                        row.append('%d'%(best_epoch))
                        rows.append(row)

        f = open(os.path.join(TABLE_DIR, '11_9_23-%s-table.csv'%(meow)), 'w')
        for row in rows:
            f.write(','.join(row) + '\n')

        f.close()


if __name__ == '__main__':
    tablify_results_11_9_23()
