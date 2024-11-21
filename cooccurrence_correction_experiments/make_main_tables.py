import os
import sys
import pandas as pd
import pickle
import re
from itertools import product
import numpy as np
from tqdm import tqdm


LLM_GENERATES_EVERYTHING = False
PAIR_ONLY = False
ALL_PAIRS = False
CONJUNCTION_OR = False
CONJUNCTION_WITH = False
CONJUNCTION_NEXT_TO = False
CONJUNCTION_NOT = False
ALL_CONJUNCTIONS = True
assert(int(LLM_GENERATES_EVERYTHING) + int(PAIR_ONLY) + int(ALL_PAIRS) + int(CONJUNCTION_OR) + int(CONJUNCTION_WITH) + int(CONJUNCTION_NEXT_TO) + int(CONJUNCTION_NOT) + int(ALL_CONJUNCTIONS) <= 1)
ABLATION_SUFFIX = ''
if LLM_GENERATES_EVERYTHING:
    ABLATION_SUFFIX = '/LLM_GENERATES_EVERYTHING'
if PAIR_ONLY:
    ABLATION_SUFFIX = '/PAIR_ONLY'
if ALL_PAIRS:
    ABLATION_SUFFIX = '/ALL_PAIRS'
if CONJUNCTION_OR:
    ABLATION_SUFFIX = '/CONJUNCTION_OR'
if CONJUNCTION_WITH:
    ABLATION_SUFFIX = '/CONJUNCTION_WITH'
if CONJUNCTION_NEXT_TO:
    ABLATION_SUFFIX = '/CONJUNCTION_NEXT_TO'
if CONJUNCTION_NOT:
    ABLATION_SUFFIX = '/CONJUNCTION_NOT'
if ALL_CONJUNCTIONS:
    ABLATION_SUFFIX = '/ALL_CONJUNCTIONS'

BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
DATASET_NAME_DISP_LIST = ['COCO', 'VOC', 'NUSWIDE']
DATASET_NAME_DICT = {'COCO' : 'COCO2014_partial', 'VOC' : 'VOC2007_partial', 'NUSWIDE' : 'nuswideTheirVersion_partial'}
MODEL_TYPE_DISP_LIST = ['ViT-L/14336px', 'ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50x64', 'RN50x16', 'RN50x4', 'RN101', 'RN50']
BASELINE_SUPERMETHOD_LIST = ['baseline']
BASELINE_SUBMETHOD_LIST = {0 : ['ZSCLIP', 'ZSCLIP (debiased)'], 1 : ['-']}
OUR_SUPERMETHOD_LIST = {0 : ['comp only', 'uniform', 'PCA'], 1 : ['uniform', 'PCA']}
OUR_SUPERMETHOD_DICT = {'comp only' : '', 'uniform' : '_avg', 'PCA' : '_pca'}
#OUR_SUBMETHOD_LIST = {0 : ['%dmax'%(topk) for topk in range(1,8)] + ['p50', 'mean'] + ['mean$>=%d$'%(topk) for topk in range(2,8)], 1 : ['mean$>=3$']}
OUR_SUBMETHOD_LIST = {0 : ['1max', '2max', '3max', '4max', '5max', '6max', '7max', 'mean', 'mean$>=2$', 'mean$>=3$', 'mean$>=4$', 'mean$>=5$', 'mean$>=6$', 'mean$>=7$', 'allpcawsing', 'allpcawsingafter1', 'allpcawsingafter2', 'allpcawsingafter3', 'allpcawsingafter4', 'allpcawsingafter5'], 1 : ['mean$>=3$', 'allpca', 'allpcawsing']}
OUR_SUBMETHOD_DICT = {'1max' : 'first_max', '2max' : 'second_max', '3max' : 'third_max', '4max' : 'fourth_max', '5max' : 'fifth_max', '6max' : 'sixth_max', '7max' : 'seventh_max', 'p25' : 'p25compounds', 'p50' : 'p50compounds', 'p75' : 'p75compounds', 'mean' : 'meancompounds', 'IQRmean' : 'IQRmeancompounds', 'mean$>=2$' : 'meanafter1compounds', 'mean$>=3$' : 'meanafter2compounds', 'mean$>=4$' : 'meanafter3compounds', 'mean$>=5$' : 'meanafter4compounds', 'mean$>=6$' : 'meanafter5compounds', 'mean$>=7$' : 'meanafter6compounds', 'allpca' : 'allpca', 'allpcawsing' : 'allpcawsing', 'allpcawsingafter1' : 'allpcawsingafter1', 'allpcawsingafter2' : 'allpcawsingafter2', 'allpcawsingafter3' : 'allpcawsingafter3', 'allpcawsingafter4' : 'allpcawsingafter4', 'allpcawsingafter5' : 'allpcawsingafter5'}
UNDERLINE_THRESHOLD = 0.2
#MAIN_TABLES_FILENAME = {0 : os.path.join(BASE_DIR, 'main_tables.tex'), 1 : os.path.join(BASE_DIR, 'main_tables_tersified.tex')}
MAIN_TABLES_FILENAME = {0 : os.path.join(BASE_DIR, 'main_tables_pcaidea%s.tex'%(ABLATION_SUFFIX.replace('/', '_'))), 1 : os.path.join(BASE_DIR, 'main_tables_tersified.tex')}


def get_baseline_method(supermethod, submethod):
    assert(supermethod == 'baseline')
    if submethod in ['ZSCLIP', '-']:
        return 'ensemble_single_uncalibrated'
    elif submethod == 'ZSCLIP (debiased)':
        return 'ensemble_single_calibrated'
    else:
        assert(False)


def get_our_method(supermethod, submethod):
    assert(supermethod in ['comp only', 'uniform', 'PCA'])
    return OUR_SUBMETHOD_DICT[submethod] + OUR_SUPERMETHOD_DICT[supermethod]


def get_method(supermethod, submethod):
    if supermethod == 'baseline':
        return get_baseline_method(supermethod, submethod)
    else:
        return get_our_method(supermethod, submethod)


def load_results(dataset_name, model_type):
    result_filename = os.path.join(BASE_DIR, '%s_test'%(dataset_name.split('_')[0]) + ABLATION_SUFFIX + '/result_files/%s_test_%s_results.pkl'%(dataset_name.split('_')[0], model_type))
    with open(result_filename, 'rb') as f:
        result_dict = pickle.load(f)

    return result_dict


#return bold_mask, underline_mask
def make_sota_mask(data, tersify=0):
    sota = np.amax(data, axis=0, keepdims=True)
    bold_mask = (data == sota)
    if tersify:
        return bold_mask, np.zeros_like(bold_mask)

    underline_mask = ((~bold_mask) & (data >= sota - UNDERLINE_THRESHOLD))
    return bold_mask, underline_mask


def format_cell(v, bold, underline):
    assert(not (bold and underline))
    if bold:
        return f"\\textbf{{{v:.1f}}}"
    elif underline:
        return f"\\underline{{{v:.1f}}}"
    else:
        return f"{v:.1f}"


#this will do bolding and underlining alongside rounding
def format_data(data, bold_mask, underline_mask):
    return [[format_cell(v, bold, underline) for v, bold, underline in zip(data_row, bold_row, underline_row)] for data_row, bold_row, underline_row in zip(data, bold_mask, underline_mask)]


def replace_tabular(match):
    l_count = len(match.group(1))
    assert(l_count >= 4)
    middle_cs = 'c' * (l_count - 3)
    return f"\\begin{{tabular}}{{ll{middle_cs}|c}}"


#return latex_code, avg_dict
def make_main_table_onedataset(dataset_name_disp, tersify=0):
    dataset_name = DATASET_NAME_DICT[dataset_name_disp]
    columns = pd.MultiIndex.from_product([[dataset_name_disp], MODEL_TYPE_DISP_LIST + ['Avg (archs)']])
    row_tuples = list(product(BASELINE_SUPERMETHOD_LIST,BASELINE_SUBMETHOD_LIST[tersify])) + list(product(OUR_SUPERMETHOD_LIST[tersify],OUR_SUBMETHOD_LIST[tersify]))
    index = pd.MultiIndex.from_tuples(row_tuples)
    methods = [get_method(*t) for t in row_tuples]
    data = [] #as list of columns
    for model_type_disp in tqdm(MODEL_TYPE_DISP_LIST):
        model_type = model_type_disp.replace('/', '')
        result_dict = load_results(dataset_name, model_type)
        data.append([result_dict[method]['mAP'] for method in methods])

    data = np.array(data).T
    avg = np.mean(data, axis=1, keepdims=True)
    data = np.hstack([data, avg])
    bold_mask, underline_mask = make_sota_mask(data, tersify=tersify)
    data = format_data(data, bold_mask, underline_mask)
    df = pd.DataFrame(data, index=index, columns=columns)
    latex_code = df.to_latex(multicolumn=True, multirow=True, escape=False)
    latex_code = re.sub(r"(\\multicolumn\{\d+\})\{l\}", r"\1{c}", latex_code)
    latex_code = re.sub(r"\\begin\{tabular\}\{(l+)\}", replace_tabular, latex_code)
    latex_code = latex_code.replace('comp only', '\\makecell{comp \\\\ only}').replace('ViT-L/14336px', '\\makecell{ViT-L/14 \\\\ 336px}').replace('Avg (archs)', '\\makecell{Avg \\\\ (archs)}')
    for x_factor in ['64', '16', '4']:
        latex_code = latex_code.replace('RN50x' + x_factor, '\\makecell{RN50 \\\\ x' + x_factor + '}')

    latex_code = '\\begin{table*}[ht]\n\\centering\n' + latex_code + '\\caption{\\captionMainTable%s}\n\\label{tab:MainTable%s}\n\\end{table*}'%(dataset_name_disp, dataset_name_disp)
    return latex_code, {method : v for method, v in zip(methods, np.squeeze(avg))}


def make_main_table_avg(avg_dict_dict, tersify=0):
    columns = pd.MultiIndex.from_tuples([(dataset_name_disp, 'Avg (archs)') for dataset_name_disp in DATASET_NAME_DISP_LIST] + [('-', 'Avg (datasets)')])
    row_tuples = list(product(BASELINE_SUPERMETHOD_LIST,BASELINE_SUBMETHOD_LIST[tersify])) + list(product(OUR_SUPERMETHOD_LIST[tersify],OUR_SUBMETHOD_LIST[tersify]))
    index = pd.MultiIndex.from_tuples(row_tuples)
    methods = [get_method(*t) for t in row_tuples]
    data = np.array([[avg_dict_dict[dataset_name_disp][method] for method in methods] for dataset_name_disp in DATASET_NAME_DISP_LIST]).T
    avg = np.mean(data, axis=1, keepdims=True)
    data = np.hstack([data, avg])
    bold_mask, underline_mask = make_sota_mask(data, tersify=tersify)
    data = format_data(data, bold_mask, underline_mask)
    df = pd.DataFrame(data, index=index, columns=columns)
    latex_code = df.to_latex(multicolumn=True, multirow=True, escape=False)
    latex_code = re.sub(r"(\\multicolumn\{\d+\})\{l\}", r"\1{c}", latex_code)
    latex_code = re.sub(r"\\begin\{tabular\}\{(l+)\}", replace_tabular, latex_code)
    latex_code = latex_code.replace('comp only', '\\makecell{comp \\\\ only}').replace('Avg (archs)', '\\makecell{Avg \\\\ (archs)}').replace('Avg (datasets)', '\\makecell{Avg \\\\ (datasets)}')
    latex_code = '\\begin{table*}[ht]\n\\centering\n' + latex_code + '\\caption{\\captionMainTableAll}\n\\label{tab:MainTableAll}\n\\end{table*}'
    return latex_code


def make_main_tables(tersify=0):
    tersify = int(tersify)
    latex_codes = []
    avg_dict_dict = {}
    for dataset_name_disp in DATASET_NAME_DISP_LIST:
        latex_code, avg_dict = make_main_table_onedataset(dataset_name_disp, tersify=tersify)
        latex_codes.append(latex_code)
        avg_dict_dict[dataset_name_disp] = avg_dict

    latex_code = make_main_table_avg(avg_dict_dict, tersify=tersify)
    latex_codes.append(latex_code)
    latex_code = '\n\n'.join(latex_codes)
    f = open(MAIN_TABLES_FILENAME[tersify], 'w')
    f.write(latex_code)
    f.close()


def usage():
    print('Usage: python make_main_tables.py [<tersify>=0]')


if __name__ == '__main__':
    make_main_tables(*(sys.argv[1:]))
