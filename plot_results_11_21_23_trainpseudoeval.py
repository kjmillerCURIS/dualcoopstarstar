import os
import sys
from matplotlib import pyplot as plt
from result_utils_11_21_23 import extract_mAPs_given_hparams_trainpseudoeval, extract_mAPs_given_runID, extract_random_chance_mAPs, extract_zsclip_mAPs

#def extract_mAPs_given_hparams(prompt_mode,num_encoder_layers,num_decoder_layers,spatial_lower,spatial_upper,temperature_is_learnable):


COLORS = ['r', 'g', 'b', 'y', 'm', 'c', 'gray', 'orange', 'olive', 'darkgoldenrod']
PLOT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/plots_and_tables')


#(prompt_mode, spatial_lower, spatial_upper, temperature_is_learnable) ==> color
def build_color_dict():
    color_dict = {}
    t = 0
    for do_adjust_logits in [1]: #[0,1]
        for use_bias in [0,1]:
            for stepsize in [0.0625, 0.125, 0.25, 0.5, 1.0]:
                color_dict[(do_adjust_logits, use_bias, stepsize)] = COLORS[t]
                t += 1

    return color_dict


def build_linestyle_dict():
    linestyle_dict = {}
    linestyle_dict[0.1] = 'dotted'
    linestyle_dict[0.2] = 'dashed'
    linestyle_dict[0.4] = 'solid'
    return linestyle_dict


#returns color, linestyle
def hparams_to_color_linestyle(do_adjust_logits, use_bias, bandwidth, stepsize):
    color_dict = build_color_dict()
    linestyle_dict = build_linestyle_dict()
    color = color_dict[(do_adjust_logits, use_bias, stepsize)]
    linestyle = linestyle_dict[bandwidth]
    return color, linestyle


def plot_results_11_21_23_trainpseudoeval():
    plt.clf()
    plt.figure(figsize=[16,12])
    for do_adjust_logits in [1]: #[0,1]
        for use_bias in [0,1]:
            for bandwidth in [0.1, 0.2, 0.4]:
                for stepsize in [0.0625, 0.125, 0.25, 0.5, 1.0]:
                    color, linestyle = hparams_to_color_linestyle(do_adjust_logits, use_bias, bandwidth, stepsize)
                    ret = extract_mAPs_given_hparams_trainpseudoeval(do_adjust_logits, use_bias, bandwidth, stepsize)
                    if ret is None:
                        continue

                    label = 'adjust_logits=%d, use_bias=%d, bandwidth=%.5f, stepsize=%.2f'%(do_adjust_logits,use_bias,bandwidth,stepsize)
                    plt.plot(ret['epoch'], ret['mAP'], color=color, linestyle=linestyle, label=label)

    plt.ylim((55, 70))
    plt.xlabel('epoch')
    plt.ylabel('training pseudolabel mAP (%)')
    plt.title('training pseudolabel mAP')
    plot_filename = os.path.join(PLOT_DIR, '11_21_23-trainpseudoeval-plot.png')
    legend_filename = os.path.join(PLOT_DIR, '11_21_23-trainpseudoeval-LEGEND.png')
    plt.savefig(plot_filename)
    plt.legend()
    plt.savefig(legend_filename)
    plt.clf()


if __name__ == '__main__':
    plot_results_11_21_23_trainpseudoeval()
