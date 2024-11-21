import os
import sys
import pandas as pd
import pickle
import re
from itertools import product
import numpy as np
from tqdm import tqdm
from make_competition_tables import average_result_dicts


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
DATASET_NAME_DICT = {'COCO' : 'COCO2014_partial', 'VOC' : 'VOC2007_partial', 'NUSWIDE' : 'nuswideTheirVersion_partial'}



def load_results(dataset_name, model_type, is_calib):
    calib_suffix = ('' if is_calib else '_without_calibration')
    result_filename = os.path.join(BASE_DIR, '%s_test/result_files%s/%s_test_%s_results.pkl'%(dataset_name.split('_')[0], calib_suffix, dataset_name.split('_')[0], model_type))
    with open(result_filename, 'rb') as f:
        result_dict = pickle.load(f)

    return result_dict


def load_avg_results(dataset_name, is_calib):
    result_dicts = []
    for model_type in ['ViT-L14336px', 'ViT-L14', 'ViT-B16', 'ViT-B32', 'RN50x64', 'RN50x16', 'RN50x4', 'RN101', 'RN50']:
        one_result_dict = load_results(dataset_name, model_type, is_calib)
        result_dicts.append(one_result_dict)

    return average_result_dicts(result_dicts)


def make_end_of_row(avg_dict_dict, method):
    scores = []
    for dataset_name_disp in ['COCO', 'VOC', 'NUSWIDE']:
        scores.append(avg_dict_dict[dataset_name_disp][method]['mAP'])

    scores.append(np.mean(scores))
    return ['%.1f'%(score) for score in scores]


def make_debias_ablation_table():
    columns = ['Use compound', 'Debias', 'COCO', 'VOC', 'NUSWIDE', 'Avg']
    avg_dict_dict_uncalib = {}
    avg_dict_dict_calib = {}
    for dataset_name_disp in ['COCO', 'VOC', 'NUSWIDE']:
        avg_dict_dict_uncalib[dataset_name_disp] = load_avg_results(DATASET_NAME_DICT[dataset_name_disp], 0)
        avg_dict_dict_calib[dataset_name_disp] = load_avg_results(DATASET_NAME_DICT[dataset_name_disp], 1)

    data = []
    data.append(['', ''] + make_end_of_row(avg_dict_dict_uncalib, 'ensemble_single_uncalibrated'))
    data.append(['', '$\\checkmark$'] + make_end_of_row(avg_dict_dict_calib, 'ensemble_single_calibrated'))
    data.append(['$\\checkmark$', ''] + make_end_of_row(avg_dict_dict_uncalib, 'allpcawsing_avg'))
    data.append(['$\\checkmark$', '$\\checkmark$'] + make_end_of_row(avg_dict_dict_calib, 'allpcawsing_avg'))
    df = pd.DataFrame(data, columns=columns)
    latex_code = df.to_latex(escape=False, index=False)
    latex_code = '\\begin{table*}[ht]\n\\centering\n' + latex_code + '\\caption{\\captionDebiasAblationTable}\n\\label{tab:DebiasAblationTable}\n\\end{table*}'
    f = open(os.path.join(BASE_DIR, 'debias_ablation_table.tex'), 'w')
    f.write(latex_code)
    f.close()


def usage():
    print('Usage: python make_debias_ablation_table.py')


if __name__ == '__main__':
    make_debias_ablation_table()
