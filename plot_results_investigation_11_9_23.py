import os
import sys
from matplotlib import pyplot as plt
from result_utils_11_9_23 import extract_mAPs_given_hparams, extract_mAPs_given_runID, extract_random_chance_mAPs

#def extract_mAPs_given_hparams(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable):


COLORS = ['r', 'g', 'b', 'y', 'm', 'c', 'gray', 'orange']
PLOT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/plots_and_tables')


#(prompt_mode, spatial_lower, spatial_upper, temperature_is_learnable) ==> color
def build_color_dict():
    color_dict = {}
    t = 0
    for prompt_mode in ['pos_and_neg_learnable_prompt', 'pos_only_fixed_prompt']:
        for spatial in [0,1]:
            for temperature_is_learnable in [0,1]:
                color_dict[(prompt_mode, spatial, spatial, temperature_is_learnable)] = COLORS[t]
                t += 1

    return color_dict


def build_linestyle_dict():
    linestyle_dict = {}
    linestyle_dict[(0,1)] = 'dotted'
    linestyle_dict[(0,2)] = 'dashed'
    linestyle_dict[(2,2)] = 'solid'
    return linestyle_dict


#returns color, linestyle
#(num_encoder_layers, num_decoder_layers) ==> linestyle
#(everything else) ==> color
def hparams_to_color_linestyle(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable):
    color_dict = build_color_dict()
    linestyle_dict = build_linestyle_dict()
    color = color_dict[(prompt_mode, spatial_lower, spatial_upper, temperature_is_learnable)]
    linestyle = linestyle_dict[(num_encoder_layers, num_decoder_layers)]
    return color, linestyle


def plot_results_investigation_11_9_23():
    for meow in ['hungarian', 'max']:
        plt.clf()
        plt.figure(figsize=[16,12])

        #DualCoOp++
        ret_baseline = extract_mAPs_given_runID('baseline_coco2014_partial_tricoop_wta_soft_448_CSC_p0_5-pos200-ctx21_norm', is_baseline=True, results_dir_suffix='Caption_tri_wta_soft/rn101/nctx21_cscTrue_ctpend/seed1/results')
        plt.plot(ret_baseline['epoch'], ret_baseline['mAP'], color='k', linestyle='solid', label='DualCoOp++ (baseline)')

        #DualCoOp--
        ret_dualcoopminusminus = extract_mAPs_given_runID('cheat_detrcheat_copies_of_clip_query_hungariancheat_fixed_match_diagonal_loss_dualcoopstarstar_fullpartial_coco_p05_seed1_posneglearn_enc0_dec1_spatl1_spatu1_templearn0_mean_except_class')
        ret_dualcoopminusminus = ret_dualcoopminusminus[meow]
        plt.plot(ret_dualcoopminusminus['epoch'], ret_dualcoopminusminus['mAP'], color='k', linestyle='dashed', label='"DualCoOp--"')

        #Learnable queries, diagonal loss
        ret_lqdl = extract_mAPs_given_runID('cheat_detrcheat_none_hungariancheat_fixed_match_diagonal_loss_dualcoopstarstar_fullpartial_coco_p05_seed1_posneglearn_enc0_dec1_spatl1_spatu1_templearn0_mean_except_class')
        ret_lqdl = ret_lqdl[meow]
        plt.plot(ret_lqdl['epoch'], ret_lqdl['mAP'], color='r', linestyle='dashed', label='Learnable queries, diagonal loss')

        #Learnable queries, full loss
        ret_lqfl = extract_mAPs_given_runID('cheat_detrcheat_none_hungariancheat_fixed_match_full_loss_dualcoopstarstar_fullpartial_coco_p05_seed1_posneglearn_enc0_dec1_spatl1_spatu1_templearn0_mean_except_class')
        ret_lqfl = ret_lqfl[meow]
        plt.plot(ret_lqfl['epoch'], ret_lqfl['mAP'], color='b', linestyle='dashed', label='Learnable queries, full loss')

        num_encoder_layers, num_decoder_layers, prompt_mode, spatial, temperature_is_learnable = 0, 1, 'pos_only_fixed_prompt', 1, 1
        label = '%d enc layers, %d dec layers, %s, %s spatial embedding, %s temp'%(num_encoder_layers, num_decoder_layers, prompt_mode, {0 : 'no', 1 : 'yes'}[spatial], {0 : 'fixed', 1 : 'learnable'}[temperature_is_learnable])
        color, linestyle = hparams_to_color_linestyle(prompt_mode,num_encoder_layers,num_decoder_layers,spatial,spatial,temperature_is_learnable)
        ret = extract_mAPs_given_hparams(prompt_mode,num_encoder_layers,num_decoder_layers,spatial,spatial,temperature_is_learnable)
        ret = ret[meow]
        plt.plot(ret['epoch'], ret['mAP'], color=color, linestyle=linestyle, label=label)
        random_chance_mAP = extract_random_chance_mAPs()[meow]
        plt.plot(plt.xlim(), [random_chance_mAP, random_chance_mAP], color='k', linestyle='dotted', label='chance')
        plt.xlabel('epoch')
        plt.ylabel('mAP (%)')
        plt.title('("%s" inference strategy)'%(meow))
        plot_filename = os.path.join(PLOT_DIR, '11_9_23_investigation-%s-plot.png'%(meow))
        legend_filename = os.path.join(PLOT_DIR, '11_9_23_investigation-%s-LEGEND.png'%(meow))
        plt.savefig(plot_filename)
        plt.legend()
        plt.savefig(legend_filename)
        plt.clf()


if __name__ == '__main__':
    plot_results_investigation_11_9_23()
