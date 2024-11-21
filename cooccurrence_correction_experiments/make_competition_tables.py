import os
import sys
import pandas as pd
import pickle
import re
from itertools import product
import numpy as np
from tqdm import tqdm
from make_main_tables import OUR_SUBMETHOD_LIST, OUR_SUBMETHOD_DICT, make_sota_mask


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
DATASET_NAME_DICT = {'COCO' : 'COCO2014_partial', 'VOC' : 'VOC2007_partial', 'NUSWIDE' : 'nuswideTheirVersion_partial'}
TAGCLIP_DATASET_NAME_DISP_LIST = ['COCO', 'VOC']
TAIDPT_DATASET_NAME_DISP_LIST = ['COCO', 'VOC', 'NUSWIDE']
COMC_DATASET_NAME_DISP_LIST = ['COCO']
TAGCLIP_MODEL_TYPE_DISP_LIST = ['ViT-B/16']
TAIDPT_MODEL_TYPE_DISP_LIST = ['RN50', 'RN50*']
COMC_MODEL_TYPE_DISP_LIST = ['RN50', 'RN50*']
TAIDPT_NUM_SEEDS = 3
COMC_NUM_SEEDS = 1
TAGCLIP_BASELINE_SUPERMETHOD_LIST = ['TagCLIP']
TAIDPT_BASELINE_SUPERMETHOD_LIST = ['TaI-DPT']
COMC_BASELINE_SUPERMETHOD_LIST = ['CoMC']
BASELINE_SUBMETHOD_LIST = {0 : ['original', 'debiased'], 1 : ['-']}
OUR_SUPERMETHOD_LIST = {k : ['uniform', 'PCA'] for k in [0,1]}
OUR_SUPERMETHOD_DICT = {'comp only' : '', 'uniform' : '_avg', 'PCA' : '_pca'}
UNDERLINE_THRESHOLD = 0.2
#COMPETITION_TABLES_FILENAME = {0 : os.path.join(BASE_DIR, 'competition_tables.tex'), 1 : os.path.join(BASE_DIR, 'competition_tables_tersified.tex')}
COMPETITION_TABLES_FILENAME = {0 : os.path.join(BASE_DIR, 'competition_tables.tex'), 1 : os.path.join(BASE_DIR, 'competition_tables_pcaidea_tersified.tex')}


def get_baseline_method(supermethod, submethod):
    if submethod in ['original', '-']:
        suffix = '_uncalibrated'
    elif submethod == 'debiased':
        suffix = '_calibrated'
    else:
        assert(False)

    return supermethod + suffix


def get_our_method(supermethod, submethod):
    assert(supermethod in ['uniform', 'PCA'])
    return OUR_SUBMETHOD_DICT[submethod] + OUR_SUPERMETHOD_DICT[supermethod]


def get_method(supermethod, submethod):
    if supermethod in ['TagCLIP', 'TaI-DPT', 'CoMC']:
        return get_baseline_method(supermethod, submethod)
    else:
        return get_our_method(supermethod, submethod)


def load_tagclip_results(dataset_name, model_type, use_log, use_rowcalibbase):
    result_filename = os.path.join(BASE_DIR, '%s_test/competition_TagCLIP/TagCLIPlog%drowcalibbase%d/result_files/%s_test_%s_results_TagCLIPlog%drowcalibbase%d.pkl'%(dataset_name.split('_')[0], use_log, use_rowcalibbase, dataset_name.split('_')[0], model_type, use_log, use_rowcalibbase))
    with open(result_filename, 'rb') as f:
        result_dict = pickle.load(f)

    return result_dict


def average_result_dicts(result_dicts):
    avg_dict = {k : [] for k in sorted(result_dicts[0].keys())}
    for one_result_dict in result_dicts:
        assert(sorted(one_result_dict.keys()) == sorted(avg_dict.keys()))
        for k in sorted(one_result_dict.keys()):
            avg_dict[k].append(one_result_dict[k]['mAP'])

    avg_dict = {k : {'mAP' : np.mean(avg_dict[k])} for k in sorted(avg_dict.keys())}
    return avg_dict


def load_taidpt_or_comc_results(is_comc, dataset_name, model_type, use_rowcalibbase, is_strong):
    competitor_name = {0 : 'TaI-DPT', 1 : 'CoMC'}[is_comc]
    num_seeds = {0 : TAIDPT_NUM_SEEDS, 1 : COMC_NUM_SEEDS}[is_comc]
    strength_str = {0 : 'weak', 1 : 'strong'}[is_strong]
    result_dicts = []
    for seed in range(1, num_seeds+1):
        result_filename = os.path.join(BASE_DIR, '%s_test/competition_%s_us_%s_them_%s/%susstrong%dthemstrong%dseed%drowcalibbase%d/result_files/%s_test_%s_results_%susstrong%dthemstrong%dseed%drowcalibbase%d.pkl'%(dataset_name.split('_')[0], competitor_name, strength_str, strength_str, competitor_name, is_strong, is_strong, seed, use_rowcalibbase, dataset_name.split('_')[0], model_type, competitor_name, is_strong, is_strong, seed, use_rowcalibbase))
        with open(result_filename, 'rb') as f:
            one_result_dict = pickle.load(f)

        result_dicts.append(one_result_dict)

    result_dict = average_result_dicts(result_dicts)
    return result_dict


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


def yesno(flag):
    return {True : 'YES', False : 'NO'}[flag]


def replace_tabular(match):
    l_count = len(match.group(1))
    assert(l_count >= 4)
    middle_cs = 'c' * (l_count - 3)
    return f"\\begin{{tabular}}{{ll{middle_cs}|c}}"


def make_competition_table_tagclip(use_log, use_rowcalibbase, tersify=0):
    title_str = {0 : 'TagCLIP (log=%s, row-calibrate baseline=%s)'%(yesno(use_log), yesno(use_rowcalibbase)), 1 : 'TagCLIP'}[tersify]
    column_tuples = list(product([title_str], TAGCLIP_DATASET_NAME_DISP_LIST, TAGCLIP_MODEL_TYPE_DISP_LIST))
    columns = pd.MultiIndex.from_tuples(column_tuples + [('-', '-', 'Avg')])
    row_tuples = list(product(TAGCLIP_BASELINE_SUPERMETHOD_LIST,BASELINE_SUBMETHOD_LIST[tersify])) + list(product(OUR_SUPERMETHOD_LIST[tersify],OUR_SUBMETHOD_LIST[tersify]))
    index = pd.MultiIndex.from_tuples(row_tuples)
    methods = [get_method(*t) for t in row_tuples]
    data = [] #as list of columns
    for column_tuple in tqdm(column_tuples):
        assert(len(column_tuple) == 3)
        dataset_name = DATASET_NAME_DICT[column_tuple[1]]
        model_type = column_tuple[2].replace('/', '')
        result_dict = load_tagclip_results(dataset_name, model_type, use_log, use_rowcalibbase)
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
    latex_code = '\\begin{table*}[ht]\n\\centering\n' + latex_code + '\\caption{\\captionCompetitionTagCLIPlog%srowcalibbase%s}\n\\label{CompetitionTagCLIPlog%srowcalibbase%s}\n\\end{table*}'%(yesno(use_log), yesno(use_rowcalibbase), yesno(use_log), yesno(use_rowcalibbase))
    return latex_code


def make_competition_table_taidpt_or_comc(is_comc, use_rowcalibbase, tersify=0):
    competitor_name = {0 : 'TaI-DPT', 1 : 'CoMC'}[is_comc]
    dataset_name_disp_list = {0 : TAIDPT_DATASET_NAME_DISP_LIST, 1 : COMC_DATASET_NAME_DISP_LIST}[is_comc]
    model_type_disp_list = {0 : TAIDPT_MODEL_TYPE_DISP_LIST, 1 : COMC_MODEL_TYPE_DISP_LIST}[is_comc]
    baseline_supermethod_list = {0 : TAIDPT_BASELINE_SUPERMETHOD_LIST, 1 : COMC_BASELINE_SUPERMETHOD_LIST}[is_comc]
    title_str = {0 : '%s (row-calibrate baseline=%s)'%(competitor_name, yesno(use_rowcalibbase)), 1 : competitor_name}[tersify]
    column_tuples = list(product([title_str], dataset_name_disp_list, model_type_disp_list))
    columns = pd.MultiIndex.from_tuples(column_tuples + [('-', '-', 'Avg')])
    row_tuples = list(product(baseline_supermethod_list,BASELINE_SUBMETHOD_LIST[tersify])) + list(product(OUR_SUPERMETHOD_LIST[tersify],OUR_SUBMETHOD_LIST[tersify]))
    index = pd.MultiIndex.from_tuples(row_tuples)
    methods = [get_method(*t) for t in row_tuples]
    data = [] #as list of columns
    for column_tuple in tqdm(column_tuples):
        assert(len(column_tuple) == 3)
        dataset_name = DATASET_NAME_DICT[column_tuple[1]]
        model_type = column_tuple[2].replace('/', '').replace('*', '')
        is_strong = (column_tuple[2][-1] == '*')
        result_dict = load_taidpt_or_comc_results(is_comc, dataset_name, model_type, use_rowcalibbase, is_strong)
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
    latex_code = '\\begin{table*}[ht]\n\\centering\n' + latex_code + '\\caption{\\captionCompetition%srowcalibbase%s}\n\\label{tab:Competition%srowcalibbase%s}\n\\end{table*}'%(competitor_name.replace('-', ''), yesno(use_rowcalibbase), competitor_name.replace('-', ''), yesno(use_rowcalibbase))
    return latex_code


def make_competition_tables(tersify=0):
    tersify = int(tersify)
    latex_codes = []
    for use_log in [0, 1]:
        for use_rowcalibbase in [0, 1]:
            if tersify and not (use_log and not use_rowcalibbase):
                continue

            latex_code = make_competition_table_tagclip(use_log, use_rowcalibbase, tersify=tersify)
            latex_codes.append(latex_code)

    for is_comc in [0, 1]:
        for use_rowcalibbase in [0, 1]:
            if tersify and not use_rowcalibbase:
                continue

            latex_code = make_competition_table_taidpt_or_comc(is_comc, use_rowcalibbase, tersify=tersify)
            latex_codes.append(latex_code)

    latex_code = '\n\n'.join(latex_codes)
    f = open(COMPETITION_TABLES_FILENAME[tersify], 'w')
    f.write(latex_code)
    f.close()


def usage():
    print('Usage: python make_competition_tables.py [<tersify>=0]')


if __name__ == '__main__':
    make_competition_tables(*(sys.argv[1:]))
