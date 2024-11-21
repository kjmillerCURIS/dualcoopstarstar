import os
import sys
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from noise_model import RESULT_DICT_FILENAME


DATASET_NAME_LIST = ['COCO2014_partial', 'VOC2007_partial', 'nuswideTheirVersion_partial']
DATASET_DISP_DICT = {'COCO2014_partial' : 'COCO', 'VOC2007_partial' : 'VOC', 'nuswideTheirVersion_partial' : 'NUSWIDE'}
CLIP_MODEL_TYPE_LIST = ['ViT-L14336px', 'ViT-L14', 'ViT-B16', 'ViT-B32', 'RN50x64', 'RN50x16', 'RN50x4', 'RN101', 'RN50']
MODEL_TYPE_LIST = ['constant_f', 'AND_only', 'OR_only', 'additive', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table']
BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')


#return list/vector
#will have mFVU for each thingy in MODEL_TYPE_LIST, plus delta value, multidelta lower quartile, and multidelta upper quartile
def tablify_noise_model_results_one_column(dataset_name, clip_model_type):
    with open(RESULT_DICT_FILENAME % (dataset_name.split('_')[0], dataset_name.split('_')[0], clip_model_type), 'rb') as f:
        d = pickle.load(f)

    data_column = [d['experiments'][model_type]['eval_dict']['mFVU'] for model_type in MODEL_TYPE_LIST]
    data_column.append(d['experiments']['OR_with_AND_bonus']['model_params']['delta'])
    data_column.append(np.percentile(d['experiments']['OR_with_AND_bonus_multidelta']['model_params']['multidelta'], 25))
    data_column.append(np.percentile(d['experiments']['OR_with_AND_bonus_multidelta']['model_params']['multidelta'], 75))
    return np.array(data_column)


def tablify_noise_model_results_one_table(dataset_name):
    data = []
    for clip_model_type in CLIP_MODEL_TYPE_LIST:
        data.append(tablify_noise_model_results_one_column(dataset_name, clip_model_type))

    data = np.array(data).T
    data = [['%.3f'%(x) for x in row] for row in data]
    columns = pd.MultiIndex.from_tuples([(DATASET_DISP_DICT[dataset_name], clip_model_type) for clip_model_type in CLIP_MODEL_TYPE_LIST])
    index = [model_type.replace('_', '-') for model_type in MODEL_TYPE_LIST] + ['delta', 'lower-quartile-multidelta', 'upper-quartile-multidelta']
    df = pd.DataFrame(data, index=index, columns=columns)
    latex_code = df.to_latex(multicolumn=True, escape=False)
    latex_code = latex_code.replace('ViT-B', 'ViT-B/').replace('ViT-L', 'ViT-L/')
    latex_code = latex_code.replace('ViT-L/14336px', '\\makecell{ViT-L/14 \\\\ 336px}').replace('OR-with-AND-bonus-multidelta', '\\makecell{OR-with-AND-bonus \\\\ multidelta}')
    for x_factor in ['64', '16', '4']:
        latex_code = latex_code.replace('RN50x' + x_factor, '\\makecell{RN50 \\\\ x' + x_factor + '}')

    latex_code = '\\begin{table*}[ht]\n\\centering\n' + latex_code + '\n\\end{table*}'
    return latex_code


def tablify_noise_model_results():
    latex_codes = []
    for dataset_name in tqdm(DATASET_NAME_LIST):
        latex_codes.append(tablify_noise_model_results_one_table(dataset_name))

    latex_code = '\n\n'.join(latex_codes)
    f = open(os.path.join(BASE_DIR, 'noise_model_tables_all.tex'), 'w')
    f.write(latex_code)
    f.close()


def usage():
    print('Usage: python tablify_noise_model_results.py')


if __name__ == '__main__':
    tablify_noise_model_results()
