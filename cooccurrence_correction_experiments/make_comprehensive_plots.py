import os
import sys
import inflect
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from make_competition_tables import BASE_DIR, average_result_dicts


DATASET_NAME_LIST = ['COCO2014_partial', 'VOC2007_partial', 'nuswideTheirVersion_partial']
MODEL_TYPE_LIST = ['ViT-L14336px', 'ViT-L14', 'ViT-B16', 'ViT-B32', 'RN50x64', 'RN50x16', 'RN50x4', 'RN101', 'RN50']
COLOR_DICT = {'kmax' : 'r', 'mean>=k' : 'b', 'mean' : 'k', 'p25' : 'g', 'p50' : 'b', 'p75' : 'm'}
BASELINE_COLOR = 'gray'
MY_ENGINE = inflect.engine()


def get_method(supermethod, submethod, index=None):
    if supermethod != '':
        supermethod = '_' + supermethod

    if submethod == 'kmax':
        return MY_ENGINE.number_to_words(MY_ENGINE.ordinal(index)) + '_max' + supermethod
    elif submethod == 'mean>=k':
        return 'meanafter%dcompounds'%(index-1) + supermethod
    else:
        return submethod + 'compounds' + supermethod


#this will give average across all models and datasets
def load_result_dict():
    result_dicts = []
    for dataset_name in DATASET_NAME_LIST:
        for model_type in MODEL_TYPE_LIST:
            result_filename = os.path.join(BASE_DIR, '%s_test/result_files/%s_test_%s_results.pkl'%(dataset_name.split('_')[0], dataset_name.split('_')[0], model_type))
            with open(result_filename, 'rb') as f:
                one_result_dict = pickle.load(f)

            result_dicts.append(one_result_dict)

    result_dict = average_result_dicts(result_dicts)
    return result_dict


def make_comprehensive_plots_one_supermethod(result_dict, supermethod, my_title, plot_filename):
    result_dict = load_result_dict()
    kmax_xs = list(range(1,8))
    meanafterk_xs = list(range(1,8))
    plt.clf()
    plt.title(my_title, fontsize=18)
    plt.xlabel('k', fontsize=18)
    plt.ylabel('mAP', fontsize=18)
    for submethod, xs in zip(['kmax', 'mean>=k'], [kmax_xs, meanafterk_xs]):
        ys = []
        for k in xs:
            if submethod == 'mean>=k' and k == 1:
                method = get_method(supermethod, 'mean')
                ys.append(result_dict[method]['mAP'])
            else:
                method = get_method(supermethod, submethod, index=k)
                ys.append(result_dict[method]['mAP'])

        plt.plot(xs, ys, linestyle='-', marker='o', color=COLOR_DICT[submethod], label=submethod)

    my_xlim = plt.xlim()
    suffix = ('' if supermethod == '' else '_' + supermethod)
    plt.plot(my_xlim, 2*[result_dict['allpcawsing%s'%(suffix)]['mAP']], linestyle='--', color='k', label='maxVariance')
    plt.plot(my_xlim, 2*[result_dict['ensemble_single_calibrated']['mAP']], linestyle='--', color='gray', label='singleton')
    plt.legend(fontsize=18)
    plt.savefig(plot_filename)
    plt.clf()


def make_comprehensive_plots():
    result_dict = load_result_dict()
    make_comprehensive_plots_one_supermethod(result_dict, '', 'Rank Fusion analysis - without merging', os.path.join(BASE_DIR, 'comprehensive_plot_compoundonly.png'))
    make_comprehensive_plots_one_supermethod(result_dict, 'avg', 'Rank Fusion analysis - with merging', os.path.join(BASE_DIR, 'comprehensive_plot_avg.png'))


if __name__ == '__main__':
    make_comprehensive_plots()
