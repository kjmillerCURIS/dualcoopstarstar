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


def plot_results_11_9_23():
    for meow in ['hungarian', 'max']:
        plt.clf()
        plt.figure(figsize=[16,12])
        ret_baseline = extract_mAPs_given_runID('baseline_coco2014_partial_tricoop_wta_soft_448_CSC_p0_5-pos200-ctx21_norm', is_baseline=True, results_dir_suffix='Caption_tri_wta_soft/rn101/nctx21_cscTrue_ctpend/seed1/results')
        plt.plot(ret_baseline['epoch'], ret_baseline['mAP'], color='k', linestyle='solid', label='DualCoOp++ (baseline)')
        for num_encoder_layers, num_decoder_layers in [(0,1),(0,2),(2,2)]:
            for prompt_mode in ['pos_and_neg_learnable_prompt', 'pos_only_fixed_prompt']:
                for spatial in [0,1]:
                    if spatial == 0:
                        continue
                    for temperature_is_learnable in [0,1]:
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
        plot_filename = os.path.join(PLOT_DIR, '11_9_23-%s-plot.png'%(meow))
        legend_filename = os.path.join(PLOT_DIR, '11_9_23-%s-LEGEND.png'%(meow))
        plt.savefig(plot_filename)
        plt.legend()
        plt.savefig(legend_filename)
        plt.clf()


if __name__ == '__main__':
    plot_results_11_9_23()
