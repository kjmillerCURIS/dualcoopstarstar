import os
import sys
import copy
import imageio
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import re
from sklearn.decomposition import PCA
from tqdm import tqdm
from compute_mAP import average_precision


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')

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

#first two args are dataset name, third is model type
ENSEMBLE_SINGLE_FILENAME = os.path.join(BASE_DIR, '%s_test' + ABLATION_SUFFIX, '%s_test_baseline_zsclip_cossims_%s_ensemble_80.pkl')
SIMPLE_SINGLE_AND_COMPOUND_FILENAME = os.path.join(BASE_DIR, '%s_test' + ABLATION_SUFFIX, '%s_test_simple_single_and_compound_cossims_%s%s.pkl')

BLENDING_XS = np.arange(21) / 20

GIF_CLASSNAMES_DICT = {'nuswide_partial' : ['airport', 'animal', 'coral', 'reflection', 'swimmers', 'sun', 'fish'], 'COCO2014_partial' : ['airplane', 'baseball glove', 'book', 'cup', 'mouse', 'spoon', 'zebra']}
BLENDING_XS_FOR_GIF = np.arange(201) / 200
NUM_BINS_NEG = 200
NUM_BINS_POS = 80
FPS = 20

TAGCLIP_DIR = os.path.join(BASE_DIR, '../../TagCLIP_outputs')
#TAGCLIP_USE_LOG = True
#TAGCLIP_USE_ROWCALIBBASE = False

TAIDPT_DIR = os.path.join(BASE_DIR, '../../TaI-DPT_outputs')
COMC_DIR = os.path.join(BASE_DIR, '../../CoMC_outputs')
#TAIDPT_OR_COMC_USE_ROWCALIBBASE = False

#COCO only
COCO_VENKATESH_CLASSNAMES_RENAMER = {'food bowl' : 'bowl'}
COCO_KEVIN_CLASSNAMES_RENAMER = {'motor bike' : 'motorcycle', 'aeroplane' : 'airplane'}

ALLOW_NO_CALIBRATION = False
SAVE_EXTRA = True

WAFFLECLIP_ABLATION = False
SIMPLE_SINGLE_AND_COMPOUND_WAFFLE_FILENAME = os.path.join(BASE_DIR, '%s_test' + ABLATION_SUFFIX, '%s_test_waffle_cossims_%s.pkl')


def depluralize(s):
    if s[-1] == 's':
        return s[:-1]
    else:
        return s


def stringmatch_to_compoundprompt(classnames, compoundprompt, is_voc=False):
    for classname in classnames:
        assert(re.sub(r'[^a-zA-Z]+', ' ', classname.lower()).strip() == classname)
        assert('  ' not in classname)

    sorted_classnames = sorted(classnames, key = lambda c: len(c), reverse=True)
    matches = []
    substrate = compoundprompt.lower()
    if is_voc:
        substrate = substrate.replace('potted plant', 'pottedplant').replace('dining table', 'diningtable').replace('tv monitor', 'tvmonitor')

    substrate = ' ' + re.sub(r'[^a-zA-Z]+', ' ', substrate) + ' '
    for classname in sorted_classnames:
        flag = False
        for c in sorted(set([classname, depluralize(classname), classname + 's', classname + 'es', classname.replace('person', 'people').replace('child', 'children').replace('man', 'men').replace('foot', 'feet').replace('goose', 'geese').replace('mouse', 'mice').replace('die', 'dice').replace('tooth', 'teeth').replace('louse', 'lice').replace('leaf', 'leaves').replace('wolf', 'wolves').replace('knife', 'knives').replace('cactus', 'cacti').replace('shelf', 'shelves').replace('calf', 'calves')])):
            if ' ' + c + ' ' in substrate:
                flag = True
                substrate = substrate.replace(' ' + c + ' ', ' ! ')

        if flag:
            matches.append(classname)

    return matches


def check_classname2compoundprompts(classname2compoundprompts, classnames):
    for classname in sorted(classnames):
        if len(classname2compoundprompts[classname]) < 4:
            print('CAUTION: "%s": %s'%(classname, ', '.join(['"%s"'%(p) for p in classname2compoundprompts[classname]])))


def load_tagclip_data(dataset_name, model_type, d_ensemble_single, tagclip_use_log):
    assert(dataset_name in ['COCO2014_partial', 'VOC2007_partial'])
    assert(model_type in ['ViT-B16'])
    with open(os.path.join(TAGCLIP_DIR, '%s_test_%s_TagCLIP_outputs.pkl'%({'COCO2014_partial' : 'coco2014', 'VOC2007_partial' : 'voc2007'}[dataset_name], {'ViT-B16' : 'ViT-B-16'}[model_type])), 'rb') as f:
        d_tagclip = pickle.load(f)

    assert(len(d_tagclip['image_list']) == len(d_ensemble_single['impaths']))
    assert(all([y == os.path.basename(x) for x, y in zip(d_ensemble_single['impaths'], d_tagclip['image_list'])]))

    #VOC2007 is known to have slight mismatch in labels for some reason
    #(we got our version from TaI-DPT, no modifications on our end)
    if dataset_name == 'VOC2007_partial':
        assert(np.mean(d_tagclip['labels'].numpy() != d_ensemble_single['gts']) < 0.01)
    else:
        assert(np.all(d_tagclip['labels'].numpy() == d_ensemble_single['gts']))

    d_tagclip_out = {'classnames' : d_ensemble_single['classnames'], 'gts' : d_ensemble_single['gts'], 'impaths' : d_ensemble_single['impaths']}
    if tagclip_use_log:
        d_tagclip_out['cossims'] = np.log(d_tagclip['predictions'].numpy())
    else:
        d_tagclip_out['cossims'] = d_tagclip['predictions'].numpy()

    return d_tagclip_out


def load_taidpt_data(dataset_name, model_type, taidpt_is_actually_comc, taidpt_them_strong, taidpt_seed, d_ensemble_single):
    assert(dataset_name in ['COCO2014_partial', 'nuswideTheirVersion_partial', 'VOC2007_partial'])
    assert(model_type in ['RN50'])
    suffix = ''
    if taidpt_them_strong:
        suffix = '_strong'

    my_dir = (COMC_DIR if taidpt_is_actually_comc else TAIDPT_DIR)
    taidpt_name = ('CoMC' if taidpt_is_actually_comc else 'TaI-DPT')
    with open(os.path.join(my_dir, '%s_test_%s_%s%s_seed%d_outputs.pkl'%(dataset_name.split('_')[0], taidpt_name, model_type, suffix, taidpt_seed)), 'rb') as f:
        d_taidpt = pickle.load(f)

    assert(len(d_taidpt['impaths']) == len(d_ensemble_single['impaths']))
    assert(d_taidpt['classnames'] == d_ensemble_single['classnames'])
    d_taidpt_out = {'classnames' : d_taidpt['classnames']}
    impath2gtrow = {os.path.basename(impath) : gtrow for impath, gtrow in zip(d_taidpt['impaths'], d_taidpt['gts'])}
    impath2cossimrow = {os.path.basename(impath) : cossimrow for impath, cossimrow in zip(d_taidpt['impaths'], d_taidpt['cossims'])}
    d_taidpt_out['gts'] = np.array([impath2gtrow[os.path.basename(impath)] for impath in d_ensemble_single['impaths']])
    d_taidpt_out['cossims'] = np.array([impath2cossimrow[os.path.basename(impath)] for impath in d_ensemble_single['impaths']])
    d_taidpt_out['impaths'] = d_ensemble_single['impaths']
    d_taidpt_out['gts'] = np.maximum(d_taidpt_out['gts'], 0)
    assert(np.all(d_taidpt_out['gts'] == d_ensemble_single['gts']))
    return d_taidpt_out


#ONLY for test set!!!
def load_data(dataset_name, model_type, use_tagclip, tagclip_use_log, use_taidpt, taidpt_is_actually_comc, taidpt_us_strong, taidpt_them_strong, taidpt_seed):
    if WAFFLECLIP_ABLATION:
        assert(not (use_tagclip or use_taidpt))

    assert(dataset_name in ['VOC2007_partial', 'COCO2014_partial', 'nuswide_partial', 'nuswideTheirVersion_partial'])
    assert(not (use_tagclip and use_taidpt))
    is_voc = (dataset_name == 'VOC2007_partial')
    is_coco = (dataset_name == 'COCO2014_partial')
    if use_taidpt and (taidpt_us_strong != taidpt_them_strong):
        print((taidpt_us_strong, taidpt_them_strong))
        print('CAUTION: different image preprocessing for us vs taidpt')

    #load pkl files
    with open(ENSEMBLE_SINGLE_FILENAME % (dataset_name.split('_')[0], dataset_name.split('_')[0], model_type), 'rb') as f:
        d_ensemble_single = pickle.load(f)

    #code will treat it as ensemble80, but we know that it's actually TagCLIP
    if use_tagclip:
        d_ensemble_single = load_tagclip_data(dataset_name, model_type, d_ensemble_single, tagclip_use_log)

    #code will treat it as ensemble80, but we know that it's actually TaI-DPT
    if use_taidpt:
        d_ensemble_single = load_taidpt_data(dataset_name, model_type, taidpt_is_actually_comc, taidpt_them_strong, taidpt_seed, d_ensemble_single)

    if WAFFLECLIP_ABLATION:
        with open(SIMPLE_SINGLE_AND_COMPOUND_WAFFLE_FILENAME % (dataset_name.split('_')[0], dataset_name.split('_')[0], model_type), 'rb') as f:
            d_simple_single_and_compound = pickle.load(f)
    else:
        suffix = {False : '', True : '_for_TaI-DPT'}[use_taidpt and not taidpt_us_strong]
        with open(SIMPLE_SINGLE_AND_COMPOUND_FILENAME % (dataset_name.split('_')[0], dataset_name.split('_')[0], model_type, suffix), 'rb') as f:
            d_simple_single_and_compound = pickle.load(f)

    if is_coco:
        #class renaming
        d_ensemble_single['classnames'] = [(COCO_KEVIN_CLASSNAMES_RENAMER[c] if c in COCO_KEVIN_CLASSNAMES_RENAMER else c) for c in d_ensemble_single['classnames']]
        if not WAFFLECLIP_ABLATION:
            d_simple_single_and_compound['simple_single_and_compound_prompts'][:len(d_ensemble_single['classnames'])] = [(COCO_VENKATESH_CLASSNAMES_RENAMER[c] if c in COCO_VENKATESH_CLASSNAMES_RENAMER else c) for c in d_simple_single_and_compound['simple_single_and_compound_prompts'][:len(d_ensemble_single['classnames'])]]

    #get classnames, gts, compoundprompts, double-check everything
    classnames = d_ensemble_single['classnames']
    num_classes = len(classnames)
    if WAFFLECLIP_ABLATION:
        assert(d_simple_single_and_compound['waffle_prompts'][:num_classes] == ['A photo of a %s.'%(c) for c in classnames])
    else:
        assert(d_simple_single_and_compound['simple_single_and_compound_prompts'][:num_classes] == classnames)

    gts_arr = d_ensemble_single['gts']
    assert(np.all(d_simple_single_and_compound['gts'] == gts_arr))
    compoundprompts = d_simple_single_and_compound[('waffle_prompts' if WAFFLECLIP_ABLATION else 'simple_single_and_compound_prompts')][num_classes:]
    assert(all([' ' in p for p in compoundprompts]))

    #do string-matching stuff
    classname2compoundprompts = {c : [] for c in classnames}
    compoundprompt2classnames = {}
    for compoundprompt in tqdm(compoundprompts):
        matches = stringmatch_to_compoundprompt(classnames, compoundprompt, is_voc=is_voc)
        compoundprompt2classnames[compoundprompt] = matches
        for classname in matches:
            classname2compoundprompts[classname].append(compoundprompt)

    print(compoundprompt2classnames)
    check_classname2compoundprompts(classname2compoundprompts, classnames)

    #get cossims, make structures
    ensemble_single_cossims_arr = d_ensemble_single['cossims']
    simple_single_cossims_arr = d_simple_single_and_compound['cossims'][:,:num_classes]
    compound_cossims_arr = d_simple_single_and_compound['cossims'][:,num_classes:]
    gts = [{c : v for c, v in zip(classnames, gts_row)} for gts_row in gts_arr]
    ensemble_single_cossims = [{c : v for c, v in zip(classnames, cossims_row)} for cossims_row in ensemble_single_cossims_arr]
    simple_single_cossims = [{c : v for c, v in zip(classnames, cossims_row)} for cossims_row in simple_single_cossims_arr]
    compound_cossims = [{p : v for p, v in zip(compoundprompts, cossims_row)} for cossims_row in compound_cossims_arr]

    #return everything
    return gts, ensemble_single_cossims, simple_single_cossims, compound_cossims, classname2compoundprompts, compoundprompt2classnames, d_ensemble_single['impaths']


def calibrate_by_row(ensemble_single_cossims, simple_single_cossims, compound_cossims, skip_rowcalibbase=False):
    ensemble_single_cossims_calib = []
    compound_cossims_calib = []
    for ensemble_single_row, simple_single_row, compound_row in zip(ensemble_single_cossims, simple_single_cossims, compound_cossims):
        ensemble_mean = np.mean([ensemble_single_row[k] for k in sorted(ensemble_single_row.keys())])
        ensemble_sd = np.std([ensemble_single_row[k] for k in sorted(ensemble_single_row.keys())], ddof=1)
        simple_mean = np.mean([simple_single_row[k] for k in sorted(simple_single_row.keys())])
        simple_sd = np.std([simple_single_row[k] for k in sorted(simple_single_row.keys())], ddof=1)
        if skip_rowcalibbase:
            ensemble_single_row_calib = ensemble_single_row
        else:
            ensemble_single_row_calib = {k : (ensemble_single_row[k]-ensemble_mean)/ensemble_sd for k in sorted(ensemble_single_row.keys())}
        compound_row_calib = {k : (compound_row[k] - simple_mean) / simple_sd for k in sorted(compound_row.keys())}
        ensemble_single_cossims_calib.append(ensemble_single_row_calib)
        compound_cossims_calib.append(compound_row_calib)

    return ensemble_single_cossims_calib, compound_cossims_calib


def calibrate_by_column(cossims):
    my_keys = sorted(cossims[0].keys())
    cossims_arr = np.array([[cossims_row[k] for k in my_keys] for cossims_row in cossims])
    means = np.mean(cossims_arr, axis=0, keepdims=True)
    sds = np.std(cossims_arr, axis=0, keepdims=True, ddof=1)
    cossims_arr = (cossims_arr - means) / sds
    cossims = [{k : cossims_arr_row[j] for j, k in enumerate(my_keys)} for cossims_arr_row in cossims_arr]
    return cossims


#return ensemble_single_cossims, compound_cossims
#yes, these are calibrated even though I call them cossims! That's the nomenclature now! Deal with it!
#there's no "calibration type", either you call this function or you don't
def calibrate_data(ensemble_single_cossims, simple_single_cossims, compound_cossims, skip_rowcalibbase=False):
    ensemble_single_cossims, compound_cossims = calibrate_by_row(ensemble_single_cossims, simple_single_cossims, compound_cossims, skip_rowcalibbase=skip_rowcalibbase)
    ensemble_single_cossims = calibrate_by_column(ensemble_single_cossims)
    compound_cossims = calibrate_by_column(compound_cossims)
    return ensemble_single_cossims, compound_cossims


#return ensemble_single_scores, first_max_scores, second_max_scores, third_max_scores as 1D np arrays
#yes, this does the whole dataset, but just for one class
#no, this does NOT do calibration, it is your responsibility to do that beforehand if you want it!
def predict_oneclass(classname,ensemble_single_cossims,compound_cossims,classname2compoundprompts):
    assert(len(ensemble_single_cossims) == len(compound_cossims))
    sorted_compound_scores_list = []
    ensemble_single_scores = []
    first_max_scores = []
    second_max_scores = []
    third_max_scores = []
    fourth_max_scores = []
    fifth_max_scores = []
    sixth_max_scores = []
    seventh_max_scores = []
    p25_scores = []
    p50_scores = []
    p75_scores = []
    mean_scores = []
    IQRmean_scores = []
    meanafter1_scores = []
    meanafter2_scores = []
    meanafter3_scores = []
    meanafter4_scores = []
    meanafter5_scores = []
    meanafter6_scores = []
    for ensemble_single_row, compound_row in zip(ensemble_single_cossims, compound_cossims):
        ensemble_single_score = ensemble_single_row[classname]
        compound_scores = np.array([compound_row[prompt] for prompt in classname2compoundprompts[classname]])
        sorted_compound_scores = np.sort(compound_scores) #use index -topk, REMEMBER TO MAKE IT NEGATIVE!!!
        sorted_compound_scores_list.append(sorted_compound_scores)
        topk_score_list = []
        for topk in [1,2,3,4,5,6,7]:
            if len(sorted_compound_scores) < topk + 1:
                topk_score_list.append(ensemble_single_score)
            else:
                topk_score_list.append(sorted_compound_scores[-topk])
                if len(topk_score_list) > 1:
                    assert(topk_score_list[-1] <= topk_score_list[-2])

        first_max_score, second_max_score, third_max_score, fourth_max_score, fifth_max_score, sixth_max_score, seventh_max_score = topk_score_list
        ensemble_single_scores.append(ensemble_single_score)
        first_max_scores.append(first_max_score)
        second_max_scores.append(second_max_score)
        third_max_scores.append(third_max_score)
        fourth_max_scores.append(fourth_max_score)
        fifth_max_scores.append(fifth_max_score)
        sixth_max_scores.append(sixth_max_score)
        seventh_max_scores.append(seventh_max_score)
        if len(compound_scores) == 0:
            p25_scores.append(ensemble_single_score)
            p50_scores.append(ensemble_single_score)
            p75_scores.append(ensemble_single_score)
            mean_scores.append(ensemble_single_score)
            IQRmean_scores.append(ensemble_single_score)
        else:
            p25, p50, p75 = np.percentile(compound_scores, [25, 50, 75])
            p25_scores.append(p25)
            p50_scores.append(p50)
            p75_scores.append(p75)
            mean_scores.append(np.mean(compound_scores))
            if not np.any((compound_scores >= p25) & (compound_scores <= p75)):
                IQRmean_scores.append(np.mean(compound_scores))
            else:
                IQRmean_scores.append(np.mean(compound_scores[(compound_scores >= p25) & (compound_scores <= p75)]))

        meanafter_list = []
        for topk in [1,2,3,4,5,6]:
            if len(sorted_compound_scores) <= topk:
                meanafter_list.append(ensemble_single_score)
            else:
                meanafter_list.append(np.mean(sorted_compound_scores[:-topk]))

        meanafter1_score, meanafter2_score, meanafter3_score, meanafter4_score, meanafter5_score, meanafter6_score = meanafter_list
        meanafter1_scores.append(meanafter1_score)
        meanafter2_scores.append(meanafter2_score)
        meanafter3_scores.append(meanafter3_score)
        meanafter4_scores.append(meanafter4_score)
        meanafter5_scores.append(meanafter5_score)
        meanafter6_scores.append(meanafter6_score)

    if len(classname2compoundprompts[classname]) == 0:
        allpcawsing_scores = np.array(ensemble_single_scores)
        pca_direction = None
    else:
        stuff_for_PCAwsing = np.array(sorted_compound_scores_list)
        stuff_for_PCAwsing = [stuff_for_PCAwsing[:,i] for i in range(stuff_for_PCAwsing.shape[1])]
        stuff_for_PCAwsing.append(np.array(ensemble_single_scores))
        allpcawsing_scores, pca_direction, __ = do_PCA_oneclass(*stuff_for_PCAwsing, use_avg_for_sign=True)

    allpcawsing_scores_list = []
    for topk in [1,2,3,4,5]:
        if len(classname2compoundprompts[classname]) <= topk:
            allpcawsing_scores_list.append(np.array(ensemble_single_scores))
        else:
            stuff_for_PCAwsing_one = np.array(sorted_compound_scores_list)[:,:-topk]
            stuff_for_PCAwsing_one = [stuff_for_PCAwsing_one[:,i] for i in range(stuff_for_PCAwsing_one.shape[1])]
            stuff_for_PCAwsing_one.append(np.array(ensemble_single_scores))
            allpcawsing_scores_one, _, __ = do_PCA_oneclass(*stuff_for_PCAwsing_one, use_avg_for_sign=True)
            allpcawsing_scores_list.append(allpcawsing_scores_one)

    return np.array(ensemble_single_scores), np.array(first_max_scores), np.array(second_max_scores), np.array(third_max_scores), np.array(fourth_max_scores), np.array(fifth_max_scores), np.array(sixth_max_scores), np.array(seventh_max_scores), np.array(p25_scores), np.array(p50_scores), np.array(p75_scores), np.array(mean_scores), np.array(IQRmean_scores), np.array(meanafter1_scores), np.array(meanafter2_scores), np.array(meanafter3_scores), np.array(meanafter4_scores), np.array(meanafter5_scores), np.array(meanafter6_scores), allpcawsing_scores, allpcawsing_scores_list[0], allpcawsing_scores_list[1], allpcawsing_scores_list[2], allpcawsing_scores_list[3], allpcawsing_scores_list[4], np.array(sorted_compound_scores_list), pca_direction


def do_PCA_oneclass(*scores_list, use_avg_for_sign=False, normalize_direction_by_L1=True):
    assert(normalize_direction_by_L1)
    scores_arr = np.array(list(scores_list)).T
    assert(scores_arr.shape[0] > scores_arr.shape[1])
    my_pca = PCA()
    my_pca.fit(scores_arr)
    direction = my_pca.components_[0,:]
    if not (np.all(direction > 0) or np.all(direction < 0)):
        print(str(direction) + '!')

    if use_avg_for_sign:
        if np.mean(direction) < 0.0:
            direction = -1 * direction
    else:
        if direction[0] < 0.0:
            direction = -1 * direction

    if normalize_direction_by_L1:
        direction = direction / np.sum(np.fabs(direction))

    pca_scores = np.squeeze(scores_arr @ direction[:,np.newaxis]) #no need to center, it's just a constant offset
    pca_direction = direction
    pca_explvars = my_pca.explained_variance_
    return pca_scores, pca_direction, pca_explvars


#figure out how each topk rank does, for all the possible topk's
def make_rank_plot_oneclass(classname, ensemble_single_cossims, compound_cossims, gts, classname2compoundprompts, plot_dir, dataset_name, model_type, do_calibration):
    compoundprompts = classname2compoundprompts[classname]
    m = len(compoundprompts)
    plt.clf()
    plt.xlabel('topk')
    plt.ylabel('AP (%)')
    plt.title('"%s": AP vs topk (m=%d)'%(classname, m))
    if m > 0:
        gts_vec = np.array([gts_row[classname] for gts_row in gts])
        ensemble_single_scores = np.array([cossims_row[classname] for cossims_row in ensemble_single_cossims])
        ensemble_single_AP = 100.0 * average_precision(ensemble_single_scores, gts_vec)
        compound_scores = np.array([[cossims_row[p] for p in compoundprompts] for cossims_row in compound_cossims])
        compound_scores = np.sort(compound_scores, axis=1)
        topk_mean_scores = np.mean(compound_scores, axis=1)
        topk_mean_AP = 100.0 * average_precision(topk_mean_scores, gts_vec)
        topk_mean_avg_scores = 0.5 * topk_mean_scores + 0.5 * ensemble_single_scores
        topk_mean_avg_AP = 100.0 * average_precision(topk_mean_avg_scores, gts_vec)
        topk_max_APs = []
        topk_max_avg_APs = []
        topk_max_pca_APs = []
        for topk in range(1, m+1):
            topk_max_scores = compound_scores[:,-topk]
            topk_max_avg_scores = 0.5 * topk_max_scores + 0.5 * ensemble_single_scores
            topk_max_pca_scores, _, __ = do_PCA_oneclass(topk_max_scores, ensemble_single_scores)
            topk_max_APs.append(100.0 * average_precision(topk_max_scores, gts_vec))
            topk_max_avg_APs.append(100.0 * average_precision(topk_max_avg_scores, gts_vec))
            topk_max_pca_APs.append(100.0 * average_precision(topk_max_pca_scores, gts_vec))

        plt.plot([1, m], [ensemble_single_AP, ensemble_single_AP], color='gray', linestyle='--', label='ensemble80')
        plt.plot([1, m], [topk_mean_AP, topk_mean_AP], color='r', linestyle='--', label='topk_mean')
        plt.plot([1, m], [topk_mean_avg_AP, topk_mean_avg_AP], color='k', linestyle='--', label='topk_mean_avg')
        plt.plot(range(1, m+1), topk_max_APs, color='r', marker='o', label='topk_max')
        plt.plot(range(1, m+1), topk_max_avg_APs, color='k', marker='o', label='topk_max_avg')
        plt.plot(range(1, m+1), topk_max_pca_APs, color='b', marker='o', label='topk_max_pca')
        plt.legend()
        plt.xticks(range(1, m+1))

    plot_subdir = os.path.join(plot_dir, 'rank_per_class')
    os.makedirs(plot_subdir, exist_ok=True)
    plot_filename = os.path.join(plot_subdir, '%s_rank_%s_%s_calib%d.png'%(classname.replace(' ', ''), dataset_name.split('_')[0], model_type, do_calibration))
    plt.savefig(plot_filename)
    plt.clf()
    plt.close()


def compute_topk_presence(classname, compound_cossims, gts, classname2compoundprompts, compoundprompt2classnames, topk, gt_filter):
    assert(False)
    assert(topk in [1,2,3])
    assert(gt_filter in [0,1,None])
    compoundprompts = classname2compoundprompts[classname]
    numerator = 0.0
    denominator = 0.0
    for compound_row, gts_row in zip(compound_cossims, gts):
        if (gt_filter is not None) and (gts_row[classname] != gt_filter):
            continue

        denominator += 1
        thingies = [(p, compound_row[p]) for p in compoundprompts]
        sorted_thingies = sorted(thingies, key=lambda thingy: thingy[1], reverse=True)
        p_topk = sorted_thingies[topk - 1][0]
        other_classnames = [c for c in compoundprompt2classnames[p_topk] if c != classname]
        if any([gts_row[c] for c in other_classnames]):
            numerator += 1

    return numerator / denominator


def compute_topk_mean_cossim(classname, compound_cossims, gts, classname2compoundprompts, topk, gt_filter):
    assert(False)
    assert(topk in [1,2,3])
    assert(gt_filter in [0,1,None])
    compoundprompts = classname2compoundprompts[classname]
    values = []
    for compound_row, gts_row in zip(compound_cossims, gts):
        if (gt_filter is not None) and (gts_row[classname] != gt_filter):
            continue

        scores = [compound_row[p] for p in compoundprompts]
        value = np.sort(scores)[-topk]
        values.append(value)

    return np.mean(values)


#ensemble_single_scores and topk_scores should be dicts that map classname to 1D npy array
#gts should still be list of dicts
def make_topk_blending_curve(classname, ensemble_single_scores, topk_scores, gts):
    curve = []
    gts_arr = np.array([gts_row[classname] for gts_row in gts])
    for blending_x in BLENDING_XS:
        blended_scores = blending_x * topk_scores[classname] + (1 - blending_x) * ensemble_single_scores[classname]
        AP = 100.0 * average_precision(blended_scores, gts_arr)
        curve.append(AP)

    return np.array(curve)


def plot_APs(ensemble_single_APs, first_max_APs, second_max_APs, third_max_APs, first_max_avg_APs, second_max_avg_APs, third_max_avg_APs, first_max_pca_APs, second_max_pca_APs, third_max_pca_APs, plot_dir, dataset_name, model_type, do_calibration):
    os.makedirs(plot_dir, exist_ok=True)
    classnames = sorted(ensemble_single_APs.keys())
    ensemble_single_mAP = np.mean([ensemble_single_APs[classname] for classname in classnames])
    first_max_mAP = np.mean([first_max_APs[classname] for classname in classnames])
    second_max_mAP = np.mean([second_max_APs[classname] for classname in classnames])
    third_max_mAP = np.mean([third_max_APs[classname] for classname in classnames])
    first_max_avg_mAP = np.mean([first_max_avg_APs[classname] for classname in classnames])
    second_max_avg_mAP = np.mean([second_max_avg_APs[classname] for classname in classnames])
    third_max_avg_mAP = np.mean([third_max_avg_APs[classname] for classname in classnames])
    first_max_pca_mAP = np.mean([first_max_pca_APs[classname] for classname in classnames])
    second_max_pca_mAP = np.mean([second_max_pca_APs[classname] for classname in classnames])
    third_max_pca_mAP = np.mean([third_max_pca_APs[classname] for classname in classnames])
    print('ensemble_single_mAP=%f'%(ensemble_single_mAP))
    print('first_max_mAP=%f'%(first_max_mAP))
    print('second_max_mAP=%f'%(second_max_mAP))
    print('third_max_mAP=%f'%(third_max_mAP))
    print('first_max_avg_mAP=%f'%(first_max_avg_mAP))
    print('second_max_avg_mAP=%f'%(second_max_avg_mAP))
    print('third_max_avg_mAP=%f'%(third_max_avg_mAP))
    print('first_max_pca_mAP=%f'%(first_max_pca_mAP))
    print('second_max_pca_mAP=%f'%(second_max_pca_mAP))
    print('third_max_pca_mAP=%f'%(third_max_pca_mAP))

    plt.clf()
    fig, axs = plt.subplots(9, 1, figsize=(12, 9*6), sharey=True)
    x = np.arange(len(classnames))
    treatment_name_list = ['first max', 'second max', 'third max', 'first max avg', 'second max avg', 'third max avg', 'first max PCA', 'second max PCA', 'third max PCA']
    treatment_APs_list = [first_max_APs, second_max_APs, third_max_APs, first_max_avg_APs, second_max_avg_APs, third_max_avg_APs, first_max_pca_APs, second_max_pca_APs, third_max_pca_APs]
    treatment_mAP_list = [first_max_mAP, second_max_mAP, third_max_mAP, first_max_avg_mAP, second_max_avg_mAP, third_max_avg_mAP, first_max_pca_mAP, second_max_pca_mAP, third_max_pca_mAP]
    for treatment_name, treatment_APs, treatment_mAP, ax in zip(treatment_name_list, treatment_APs_list, treatment_mAP_list, axs):
        ax.set_title('%s (mAP=%.3f) vs ensemble single (mAP=%.3f)'%(treatment_name, treatment_mAP, ensemble_single_mAP))
        ax.bar(x, -np.abs([ensemble_single_APs[c] for c in classnames]), label='ensemble single', color='blue')
        ax.bar(x, np.abs([treatment_APs[c] for c in classnames]), label=treatment_name, color='green')
        ax.scatter(x, [treatment_APs[c] - ensemble_single_APs[c] for c in classnames], color='k', marker='o')
        ax.set_xticks(x)
        ax.set_xticklabels(classnames, rotation=90, ha='center')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.7)  # Vertical gridlines
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.7)  # Horizontal gridlines
        ax.set_xlabel('class')
        ax.set_ylabel('AP (%)')
        ax.legend()

    plt.tight_layout()
    plot_filename = os.path.join(plot_dir, 'APs_%s_%s_calib%d.png'%(dataset_name.split('_')[0], model_type, do_calibration))
    plt.savefig(plot_filename)
    plt.clf()


def plot_blending_curves_helper(blending_curves_list, blending_mAP_curve_list, PCA_directions_list, curve_name_list, plot_dir, dataset_name, model_type, do_calibration):
    os.makedirs(os.path.join(plot_dir, 'blending_per_class'), exist_ok=True)

    assert(len(blending_curves_list) == len(curve_name_list))
    assert(len(blending_mAP_curve_list) == len(curve_name_list))
    assert(len(PCA_directions_list) == len(curve_name_list))

    #mAP
    plt.clf()
    plt.figure(figsize=(8,12))
    for curve_name, blending_mAP_curve, PCA_directions, color in zip(curve_name_list, blending_mAP_curve_list, PCA_directions_list, ['r', 'b', 'orange', 'taupe', 'm', 'k']):
        plt.plot(BLENDING_XS, blending_mAP_curve, color=color, label=curve_name)
        my_ylim = plt.ylim()
        consensus_x = np.mean([PCA_directions[c][0] for c in sorted(PCA_directions.keys())])
        plt.plot([consensus_x, consensus_x], list(my_ylim), color=color, linestyle='--', label=curve_name + ' consensus')

    my_ylim = plt.ylim()
    plt.plot([0.5, 0.5], list(my_ylim), color='k', linestyle='--', label='0.5')
    plt.title('mAPs for different weightings')
    plt.legend()
    plt.ylabel('mAP')
    plt.xlabel('weight for max')
    plt.savefig(os.path.join(plot_dir, 'blending_mAPs_%s_%s_calib%d.png'%(dataset_name.split('_')[0], model_type, do_calibration)))
    plt.clf()

    #APs
    classnames = sorted(blending_curves_list[0].keys())
    for classname in tqdm(classnames):
        plt.clf()
        plt.figure(figsize=(8,6))
        for curve_name, blending_curves, PCA_directions, color in zip(curve_name_list, blending_curves_list, PCA_directions_list, ['r', 'b', 'orange', 'taupe', 'm', 'k']):
            plt.plot(BLENDING_XS, blending_curves[classname], color=color, label=curve_name)
            my_ylim = plt.ylim()
            consensus_x = np.mean([PCA_directions[c][0] for c in sorted(PCA_directions.keys())])
            plt.plot([consensus_x, consensus_x], list(my_ylim), color=color, linestyle='--', label=curve_name + ' consensus')
            plt.plot([PCA_directions[classname][0], PCA_directions[classname][0]], list(my_ylim), color=color, linestyle='dotted', label=curve_name + ' pca')

        my_ylim = plt.ylim()
        plt.plot([0.5, 0.5], list(my_ylim), color='k', linestyle='--', label='0.5')
        plt.title('"%s": APs for different weightings'%(classname))
        plt.legend()
        plt.ylabel('AP')
        plt.xlabel('weight for max')
        plt.savefig(os.path.join(plot_dir, 'blending_per_class', '%s_blending_mAPs_%s_%s_calib%d.png'%(classname.replace(' ', ''), dataset_name.split('_')[0], model_type, do_calibration)))
        plt.clf()


def plot_blending_curves(first_max_curves, second_max_curves, third_max_curves, first_max_mAP_curve, second_max_mAP_curve, third_max_mAP_curve, plot_dir, dataset_name, model_type, do_calibration):
    os.makedirs(os.path.join(plot_dir, 'blending_per_class'), exist_ok=True)

    #mAP
    plt.clf()
    plt.figure(figsize=(8,12))
    plt.plot(BLENDING_XS, first_max_mAP_curve, color='r', label='first max')
    plt.plot(BLENDING_XS, second_max_mAP_curve, color='k', label='second max')
    plt.plot(BLENDING_XS, third_max_mAP_curve, color='b', label='third max')
    plt.title('mAPs for different weightings')
    plt.legend()
    plt.ylabel('mAP')
    plt.xlabel('weight for max')
    plt.savefig(os.path.join(plot_dir, 'blending_mAPs_%s_%s_calib%d.png'%(dataset_name.split('_')[0], model_type, do_calibration)))
    plt.clf()

    #APs
    classnames = sorted(first_max_curves.keys())
    for classname in tqdm(classnames):
        plt.clf()
        plt.figure(figsize=(8,6))
        plt.plot(BLENDING_XS, first_max_curves[classname], color='r', label='first max')
        plt.plot(BLENDING_XS, second_max_curves[classname], color='k', label='second max')
        plt.plot(BLENDING_XS, third_max_curves[classname], color='b', label='third max')
        plt.title('"%s": APs for different weightings'%(classname))
        plt.legend()
        plt.ylabel('AP')
        plt.xlabel('weight for max')
        plt.savefig(os.path.join(plot_dir, 'blending_per_class', '%s_blending_mAPs_%s_%s_calib%d.png'%(classname.replace(' ', ''), dataset_name.split('_')[0], model_type, do_calibration)))
        plt.clf()


def plot_topk_presences(first_max_presences_given_neg, second_max_presences_given_neg, third_max_presences_given_neg, first_max_presences_given_pos, second_max_presences_given_pos, third_max_presences_given_pos, plot_dir, dataset_name, do_calibration):
    assert(False)
    os.makedirs(plot_dir, exist_ok=True)
    classnames = sorted(first_max_presences_given_neg.keys())
    bar_width = 0.25  # Width of each bar
    x = 2 * np.arange(len(classnames))  # The label locations
    plt.clf()
    plt.figure(figsize=(42, 2*8))
    plt.title('Pr(X_j | X_i) for class j in compound prompt')
    plt.ylabel('prob')
    plt.bar(x - bar_width, np.abs([first_max_presences_given_pos[c] for c in classnames]), bar_width, label='Pr(X_j | X_i=1) for j in first max', color='r')
    plt.bar(x - bar_width, -np.abs([first_max_presences_given_neg[c] for c in classnames]), bar_width, label='Pr(X_j | X_i=0) for j in first max', color='pink')
    plt.bar(x, np.abs([second_max_presences_given_pos[c] for c in classnames]), bar_width, label='Pr(X_j | X_i=1) for j in second max', color='k')
    plt.bar(x, -np.abs([second_max_presences_given_neg[c] for c in classnames]), bar_width, label='Pr(X_j | X_i=0) for j in second max', color='gray')
    plt.bar(x + bar_width, np.abs([third_max_presences_given_pos[c] for c in classnames]), bar_width, label='Pr(X_j | X_i=1) for j in third max', color='b')
    plt.bar(x + bar_width, -np.abs([third_max_presences_given_neg[c] for c in classnames]), bar_width, label='Pr(X_j | X_i=0) for j in third max', color='lightblue')
    plt.xticks(x, classnames, rotation=90, ha='center')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, axis='x', linestyle='--', color='gray', alpha=0.7)  # Vertical gridlines
    plt.grid(True, axis='y', linestyle='--', color='gray', alpha=0.7)  # Horizontal gridlines
    plt.xlabel('classname')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'topk_presence_%s_calib%d.png'%(dataset_name.split('_')[0], do_calibration)))
    plt.clf()


def plot_topk_mean_cossims(first_max_mean_cossims_given_neg, second_max_mean_cossims_given_neg, third_max_mean_cossims_given_neg, first_max_mean_cossims_given_pos, second_max_mean_cossims_given_pos, third_max_mean_cossims_given_pos, first_max_mean_cossims_given_all, second_max_mean_cossims_given_all, third_max_mean_cossims_given_all, plot_dir, dataset_name, do_calibration):
    assert(False)
    os.makedirs(plot_dir, exist_ok=True)
    classnames = sorted(first_max_mean_cossims_given_neg.keys())
    bar_width = 0.25  # Width of each bar
    x = 2 * np.arange(len(classnames))  # The label locations
    plt.clf()
    fig, axs = plt.subplots(3, 1, figsize=(42, 3*8), sharey=True)

    #all
    axs[0].set_title('E[topk score]')
    axs[0].set_ylabel('mean topk score')
    axs[0].set_xlabel('classname')
    axs[0].bar(x - bar_width, [first_max_mean_cossims_given_all[c] for c in classnames], bar_width, label='first max', color='r')
    axs[0].bar(x, [second_max_mean_cossims_given_all[c] for c in classnames], bar_width, label='second max', color='k')
    axs[0].bar(x + bar_width, [third_max_mean_cossims_given_all[c] for c in classnames], bar_width, label='third max', color='b')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(classnames, rotation=90)
    axs[0].legend()

    #pos
    axs[1].set_title('E[topk score | target class is present]')
    axs[1].set_ylabel('mean topk score')
    axs[1].set_xlabel('classname')
    axs[1].bar(x - bar_width, [first_max_mean_cossims_given_pos[c] for c in classnames], bar_width, label='first max', color='r')
    axs[1].bar(x, [second_max_mean_cossims_given_pos[c] for c in classnames], bar_width, label='second max', color='k')
    axs[1].bar(x + bar_width, [third_max_mean_cossims_given_pos[c] for c in classnames], bar_width, label='third max', color='b')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(classnames, rotation=90)
    axs[1].legend()

    #neg
    axs[2].set_title('E[topk score | target class is absent]')
    axs[2].set_ylabel('mean topk score')
    axs[2].set_xlabel('classname')
    axs[2].bar(x - bar_width, [first_max_mean_cossims_given_neg[c] for c in classnames], bar_width, label='first max', color='r')
    axs[2].bar(x, [second_max_mean_cossims_given_neg[c] for c in classnames], bar_width, label='second max', color='k')
    axs[2].bar(x + bar_width, [third_max_mean_cossims_given_neg[c] for c in classnames], bar_width, label='third max', color='b')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(classnames, rotation=90)
    axs[2].legend()


    # Tight layout for better spacing
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, 'topk_mean_score_%s_calib%d.png'%(dataset_name.split('_')[0], do_calibration)))
    plt.clf()


def plot_topk_mean_cossim_features_vs_perfdiff(first_max_mean_cossims_given_all, second_max_mean_cossims_given_all, third_max_mean_cossims_given_all, ensemble_single_APs, third_max_pca_APs, plot_dir, dataset_name, do_calibration):
    assert(False)
    plt.clf()
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    axs = axs.flatten()
    classnames = sorted(ensemble_single_APs.keys())
    perfdiffs = [third_max_pca_APs[c] - ensemble_single_APs[c] for c in classnames]
    thirds = [third_max_mean_cossims_given_all[c] for c in classnames]
    second_third_diffs = [second_max_mean_cossims_given_all[c] - third_max_mean_cossims_given_all[c] for c in classnames]
    curvatures = [first_max_mean_cossims_given_all[c] + third_max_mean_cossims_given_all[c] - 2 * second_max_mean_cossims_given_all[c] for c in classnames]
    fig.suptitle('E[1st],E[2nd],E[3rd] trend vs performance diff - each point is a class')
    axs[0].set_xlabel('E[3rdmax]')
    axs[1].set_xlabel('E[2ndmax] - E[3rdmax]')
    axs[2].set_xlabel('E[3rdmax] + E[1stmax] - 2 * E[2ndmax]')
    axs[0].set_ylabel('PCA(3rd,ensemble80) AP - ensemble80 AP')
    axs[1].set_ylabel('PCA(3rd,ensemble80) AP - ensemble80 AP')
    axs[2].set_ylabel('PCA(3rd,ensemble80) AP - ensemble80 AP')
    axs[0].scatter(thirds, perfdiffs)
    axs[1].scatter(second_third_diffs, perfdiffs)
    axs[2].scatter(curvatures, perfdiffs)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Emaxtrend_vs_perfdiff_%s_calib%d.png'%(dataset_name.split('_')[0], do_calibration)))
    plt.clf()


def make_pca_explvars_plot(baseline_APs, topk_pca_APs, topk_pca_explvars, topk_name, baseline_name, plot_dir, do_calibration, model_type, dataset_name):
    plt.clf()
    plt.title('%s PCA explained var ratio vs perfdiff (%s_pca_AP - %s_AP)'%(topk_name, topk_name, baseline_name))
    plt.xlabel('%s PCA explained var ratio'%(topk_name))
    plt.ylabel('%s PCA AP - %s AP'%(topk_name, baseline_name))
    classnames = sorted(baseline_APs.keys())
    xs = np.array([topk_pca_explvars[c][0] / np.sum(topk_pca_explvars[c]) for c in classnames])
    ys = np.array([topk_pca_APs[c] - baseline_APs[c] for c in classnames])
    thresholds = np.linspace(np.amin(xs) - 1e-6, np.amax(xs) + 1e-6, num=100, endpoint=True)
    rule_mAP_diffs = [np.sum(ys[xs > t]) / len(ys) for t in thresholds] #how much mAP would improve over baseline at each threshold
    plt.scatter(xs, ys, marker='.')
    plt.plot(thresholds, rule_mAP_diffs)
    plt.savefig(os.path.join(plot_dir, '%s_pca_explvars_vs_perfdiff_against_%s_%s_%s_calib%d.png'%(topk_name, baseline_name, dataset_name.split('_')[0], model_type, do_calibration)))
    plt.clf()


def hist_fn(scores, num_bins):
    my_hist, bin_edges = np.histogram(scores, bins=num_bins, density=True)
    assert(len(bin_edges) == len(my_hist) + 1)
    return my_hist, 0.5 * (bin_edges[:-1] + bin_edges[1:]), bin_edges


#blend_x should KEVIN
def make_one_histosweep(ensemble_single_scores, topk_scores, blend_x, gts, classname, topk_name, frame_t, plot_dir, dataset_name, do_calibration):
    assert(False)
    blended_scores = blend_x * topk_scores + (1 - blend_x) * ensemble_single_scores
    ensemble_single_AP = 100.0 * average_precision(ensemble_single_scores, gts)
    topk_AP = 100.0 * average_precision(topk_scores, gts)
    blended_AP = 100.0 * average_precision(blended_scores, gts)
    ensemble_single_scores_pos, ensemble_single_scores_neg = ensemble_single_scores[gts == 1], ensemble_single_scores[gts == 0]
    topk_scores_pos, topk_scores_neg = topk_scores[gts == 1], topk_scores[gts == 0]
    blended_scores_pos, blended_scores_neg = blended_scores[gts == 1], blended_scores[gts == 0]
    ensemble_single_hist_pos, ensemble_single_bincenters_pos, ensemble_single_binedges_pos = hist_fn(ensemble_single_scores_pos, NUM_BINS_POS)
    ensemble_single_hist_neg, ensemble_single_bincenters_neg, ensemble_single_binedges_neg = hist_fn(ensemble_single_scores_neg, NUM_BINS_NEG)
    topk_hist_pos, topk_bincenters_pos, _ = hist_fn(topk_scores_pos, NUM_BINS_POS)
    topk_hist_neg, topk_bincenters_neg, _ = hist_fn(topk_scores_neg, NUM_BINS_NEG)
    blended_hist_pos, blended_bincenters_pos, _ = hist_fn(blended_scores_pos, NUM_BINS_POS)
    blended_hist_neg, blended_bincenters_neg, _ = hist_fn(blended_scores_neg, NUM_BINS_NEG)
    should_v = True
    if frame_t < NUM_BINS_NEG:
        indices = np.nonzero((ensemble_single_scores_neg >= ensemble_single_binedges_neg[frame_t]) & (ensemble_single_scores_neg <= ensemble_single_binedges_neg[frame_t+1]))[0]
        if len(indices) == 0:
            should_v = False

        ensemble_single_mean = np.mean(ensemble_single_scores_neg[indices])
        topk_mean = np.mean(topk_scores_neg[indices])
        blended_mean = np.mean(blended_scores_neg[indices])
        mean_color = 'r'
    else:
        indices = np.nonzero((ensemble_single_scores_pos >= ensemble_single_binedges_pos[frame_t-NUM_BINS_NEG]) & (ensemble_single_scores_pos <= ensemble_single_binedges_pos[frame_t-NUM_BINS_NEG+1]))[0]
        if len(indices) == 0:
            should_v = False

        ensemble_single_mean = np.mean(ensemble_single_scores_pos[indices])
        topk_mean = np.mean(topk_scores_pos[indices])
        blended_mean = np.mean(blended_scores_pos[indices])
        mean_color = 'b'

    plt.clf()
    plt.title('"%s": ens80=%.2f, %s=%.2f, blendx=%.3f, blend=%.2f'%(classname,ensemble_single_AP,topk_name,topk_AP,blend_x,blended_AP),fontsize=8)
    plt.xlabel('score')
    plt.ylabel('density')
    plt.plot(ensemble_single_bincenters_pos, ensemble_single_hist_pos, color='b', linestyle='-', label='ensemble80 pos')
    plt.plot(ensemble_single_bincenters_neg, ensemble_single_hist_neg, color='r', linestyle='-', label='ensemble80 neg')
    plt.plot(topk_bincenters_pos, topk_hist_pos, color='b', linestyle=':', label='%s pos'%(topk_name))
    plt.plot(topk_bincenters_neg, topk_hist_neg, color='r', linestyle=':', label='%s neg'%(topk_name))
    plt.plot(blended_bincenters_pos, blended_hist_pos, color='b', linestyle='--', label='blended pos')
    plt.plot(blended_bincenters_neg, blended_hist_neg, color='r', linestyle='--', label='blended neg')
    if should_v:
        plt.axvline(x=ensemble_single_mean, color=mean_color, linestyle='-')
        plt.axvline(x=topk_mean, color=mean_color, linestyle=':')
        plt.axvline(x=blended_mean, color=mean_color, linestyle='--')

    plt.ylim((0,1))
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'histosweep_gifs', 'TEMP', 'frame%09d.png'%(frame_t)))
    plt.clf()
    plt.close()


#*_scores, gts now 1d arr
def make_histosweep_gif(ensemble_single_scores, topk_scores, blend_x, gts, classname, topk_name, plot_dir, dataset_name, do_calibration):
    assert(False)
    os.makedirs(os.path.join(plot_dir, 'histosweep_gifs', 'TEMP'), exist_ok=True)
    for frame_t in range(NUM_BINS_NEG + NUM_BINS_POS):
        make_one_histosweep(ensemble_single_scores, topk_scores, blend_x, gts, classname, topk_name, frame_t, plot_dir, dataset_name, do_calibration)

    gif_filename = os.path.join(plot_dir, 'histosweep_gifs', '%s_%s_%s_calib%d_sweep.gif'%(classname.replace(' ', ''), topk_name, dataset_name.split('_')[0], do_calibration))
    with imageio.get_writer(gif_filename, mode='I', duration=1/FPS) as writer:
        for frame_t in range(NUM_BINS_NEG + NUM_BINS_POS):
            filename = os.path.join(plot_dir, 'histosweep_gifs', 'TEMP', 'frame%09d.png'%(frame_t))
            image = imageio.imread(filename)
            writer.append_data(image)

    for frame_t in range(NUM_BINS_NEG + NUM_BINS_POS):
        filename = os.path.join(plot_dir, 'histosweep_gifs', 'TEMP', 'frame%09d.png'%(frame_t))
        os.remove(filename)

    return blend_x


def make_one_histogram(ensemble_single_scores, topk_scores, blend_x, gts, classname, topk_name, frame_t, plot_dir, dataset_name, do_calibration):
    assert(False)
    blended_scores = blend_x * topk_scores + (1 - blend_x) * ensemble_single_scores
    ensemble_single_AP = 100.0 * average_precision(ensemble_single_scores, gts)
    topk_AP = 100.0 * average_precision(topk_scores, gts)
    blended_AP = 100.0 * average_precision(blended_scores, gts)
    ensemble_single_scores_pos, ensemble_single_scores_neg = ensemble_single_scores[gts == 1], ensemble_single_scores[gts == 0]
    topk_scores_pos, topk_scores_neg = topk_scores[gts == 1], topk_scores[gts == 0]
    blended_scores_pos, blended_scores_neg = blended_scores[gts == 1], blended_scores[gts == 0]
    ensemble_single_hist_pos, ensemble_single_bincenters_pos, _ = hist_fn(ensemble_single_scores_pos, NUM_BINS_POS)
    ensemble_single_hist_neg, ensemble_single_bincenters_neg, _ = hist_fn(ensemble_single_scores_neg, NUM_BINS_NEG)
    topk_hist_pos, topk_bincenters_pos, _ = hist_fn(topk_scores_pos, NUM_BINS_POS)
    topk_hist_neg, topk_bincenters_neg, _ = hist_fn(topk_scores_neg, NUM_BINS_NEG)
    blended_hist_pos, blended_bincenters_pos, _ = hist_fn(blended_scores_pos, NUM_BINS_POS)
    blended_hist_neg, blended_bincenters_neg, _ = hist_fn(blended_scores_neg, NUM_BINS_NEG)
    plt.clf()
    plt.title('"%s": ens80=%.2f, %s=%.2f, blendx=%.3f, blend=%.2f'%(classname,ensemble_single_AP,topk_name,topk_AP,blend_x,blended_AP),fontsize=8)
    plt.xlabel('score')
    plt.ylabel('density')
    plt.plot(ensemble_single_bincenters_pos, ensemble_single_hist_pos, color='b', linestyle='-', label='ensemble80 pos')
    plt.plot(ensemble_single_bincenters_neg, ensemble_single_hist_neg, color='r', linestyle='-', label='ensemble80 neg')
    plt.plot(topk_bincenters_pos, topk_hist_pos, color='b', linestyle=':', label='%s pos'%(topk_name))
    plt.plot(topk_bincenters_neg, topk_hist_neg, color='r', linestyle=':', label='%s neg'%(topk_name))
    plt.plot(blended_bincenters_pos, blended_hist_pos, color='b', linestyle='--', label='blended pos')
    plt.plot(blended_bincenters_neg, blended_hist_neg, color='r', linestyle='--', label='blended neg')
    plt.ylim((0,1))
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'histogram_gifs', 'TEMP', 'frame%09d.png'%(frame_t)))
    plt.clf()
    plt.close()
    return blended_AP


#*_scores, gts now 1d arr
def make_histogram_gif(ensemble_single_scores, topk_scores, gts, classname, topk_name, plot_dir, dataset_name, do_calibration):
    assert(False)
    os.makedirs(os.path.join(plot_dir, 'histogram_gifs', 'TEMP'), exist_ok=True)
    best_blend_x = None
    best_blended_AP = float('-inf')
    for frame_t, blend_x in enumerate(BLENDING_XS_FOR_GIF):
        blended_AP = make_one_histogram(ensemble_single_scores, topk_scores, blend_x, gts, classname, topk_name, frame_t, plot_dir, dataset_name, do_calibration)
        if blended_AP > best_blended_AP:
            best_blended_AP = blended_AP
            best_blend_x = blend_x

    gif_filename = os.path.join(plot_dir, 'histogram_gifs', '%s_%s_%s_calib%d_hist.gif'%(classname.replace(' ', ''), topk_name, dataset_name.split('_')[0], do_calibration))
    with imageio.get_writer(gif_filename, mode='I', duration=1/FPS) as writer:
        for frame_t in range(len(BLENDING_XS_FOR_GIF)):
            filename = os.path.join(plot_dir, 'histogram_gifs', 'TEMP', 'frame%09d.png'%(frame_t))
            image = imageio.imread(filename)
            writer.append_data(image)

    for frame_t in range(len(BLENDING_XS_FOR_GIF)):
        filename = os.path.join(plot_dir, 'histogram_gifs', 'TEMP', 'frame%09d.png'%(frame_t))
        os.remove(filename)

    return best_blend_x


#*_scores should be dict that maps classname to 1d arr
#gts still list of dicts
def make_histogram_and_histosweep_gifs(ensemble_single_scores, first_max_scores, second_max_scores, third_max_scores, gts, plot_dir, dataset_name, do_calibration):
    assert(False)
    classnames = GIF_CLASSNAMES_DICT[dataset_name]
    for classname in tqdm(classnames):
        for topk_scores, topk_name in zip([first_max_scores, second_max_scores, third_max_scores], ['1stmax', '2ndmax', '3rdmax']):
            blend_x = make_histogram_gif(ensemble_single_scores[classname], topk_scores[classname], np.array([gts_row[classname] for gts_row in gts]), classname, topk_name, plot_dir, dataset_name, do_calibration)
            make_histosweep_gif(ensemble_single_scores[classname], topk_scores[classname], blend_x, np.array([gts_row[classname] for gts_row in gts]), classname, topk_name, plot_dir, dataset_name, do_calibration)


def append_to_results(method, mAP, APs, results, f):
    results[method] = {'mAP' : mAP, 'APs' : APs}
    f.write('%s,%f\n'%(method, mAP))
    print('%s: mAP=%f'%(method, mAP))


def second_max_experiments(dataset_name, model_type, do_calibration, use_tagclip=0, tagclip_use_log=0, tagclip_use_rowcalibbase=0, use_taidpt=0, taidpt_is_actually_comc=0, taidpt_or_comc_use_rowcalibbase=0, taidpt_us_strong=0, taidpt_them_strong=0, taidpt_seed=None):
    print(sys.argv)
    do_calibration = int(do_calibration)
    use_tagclip = int(use_tagclip)
    use_taidpt = int(use_taidpt)
    assert(not (use_tagclip and use_taidpt))
    if use_tagclip:
        tagclip_use_log = int(tagclip_use_log)
        tagclip_use_rowcalibbase = int(tagclip_use_rowcalibbase)
    elif use_taidpt:
        taidpt_is_actually_comc = int(taidpt_is_actually_comc)
        taidpt_or_comc_use_rowcalibbase = int(taidpt_or_comc_use_rowcalibbase)
        taidpt_us_strong = int(taidpt_us_strong)
        taidpt_them_strong = int(taidpt_them_strong)
        taidpt_seed = int(taidpt_seed)
        taidpt_name = ('CoMC' if taidpt_is_actually_comc else 'TaI-DPT')
        us_strength_str = ('strong' if taidpt_us_strong else 'weak')
        them_strength_str = ('strong' if taidpt_them_strong else 'weak')

    assert(do_calibration or ALLOW_NO_CALIBRATION)

    print('load data...')
    gts, ensemble_single_cossims, simple_single_cossims, compound_cossims, classname2compoundprompts, compoundprompt2classnames, impaths = load_data(dataset_name, model_type, use_tagclip, tagclip_use_log, use_taidpt, taidpt_is_actually_comc, taidpt_us_strong, taidpt_them_strong, taidpt_seed)
    classnames = sorted(gts[0].keys())

    #uncalibrated APs
    ensemble_single_scores_uncalib = {c : np.array([row[c] for row in ensemble_single_cossims]) for c in classnames}
    print('uncalibrated_APs...')
    ensemble_single_uncalibrated_APs = {}
    for classname in classnames:
        output = np.array([row[classname] for row in ensemble_single_cossims])
        target = np.array([row[classname] for row in gts])
        ensemble_single_uncalibrated_APs[classname] = 100.0 * average_precision(output, target)

    ensemble_single_uncalibrated_mAP = np.mean([ensemble_single_uncalibrated_APs[classname] for classname in classnames])

    #do calibration if needed
    if do_calibration:
        print('calibration...')
        skip_rowcalibbase = ((use_tagclip and not tagclip_use_rowcalibbase) or (use_taidpt and not taidpt_or_comc_use_rowcalibbase))
        ensemble_single_cossims, compound_cossims = calibrate_data(ensemble_single_cossims, simple_single_cossims, compound_cossims, skip_rowcalibbase=skip_rowcalibbase)

    plot_dir = os.path.join(BASE_DIR, dataset_name.split('_')[0] + '_test' + ABLATION_SUFFIX, model_type + '_' + {False : 'without_calibration', True : 'with_calibration'}[do_calibration])
    if use_tagclip:
        plot_dir = os.path.join(BASE_DIR, dataset_name.split('_')[0] + '_test' + ABLATION_SUFFIX, 'competition_TagCLIP', 'TagCLIPlog%drowcalibbase%d'%(tagclip_use_log, tagclip_use_rowcalibbase), model_type + '_' + {False : 'without_calibration', True : 'with_calibration'}[do_calibration])

    if use_taidpt:
        plot_dir = os.path.join(BASE_DIR, dataset_name.split('_')[0] + '_test' + ABLATION_SUFFIX, 'competition_%s_us_%s_them_%s'%(taidpt_name, us_strength_str, them_strength_str), '%susstrong%dthemstrong%dseed%drowcalibbase%d'%(taidpt_name, taidpt_us_strong, taidpt_them_strong, taidpt_seed, taidpt_or_comc_use_rowcalibbase), model_type + '_' + {False : 'without_calibration', True : 'with_calibration'}[do_calibration])

    if WAFFLECLIP_ABLATION:
        plot_dir = os.path.join(BASE_DIR, dataset_name.split('_')[0] + '_test' + ABLATION_SUFFIX, 'waffleclip_ablation', model_type + '_' + {False : 'without_calibration', True : 'with_calibration'}[do_calibration])

    os.makedirs(plot_dir, exist_ok=True)

    #rank plots
    print('rank plots...')
    for classname in tqdm(classnames):
        make_rank_plot_oneclass(classname, ensemble_single_cossims, compound_cossims, gts, classname2compoundprompts, plot_dir, dataset_name, model_type, do_calibration)

    #get the scores, each as dict mapping from classname to 1D array
    #also get the uniform averaging scores
    print('scores...')
    ensemble_single_scores = {}
    first_max_scores = {}
    second_max_scores = {}
    third_max_scores = {}
    fourth_max_scores = {}
    fifth_max_scores = {}
    sixth_max_scores = {}
    seventh_max_scores = {}
    p25_scores = {}
    p50_scores = {}
    p75_scores = {}
    mean_scores = {}
    IQRmean_scores = {}
    meanafter1_scores = {}
    meanafter2_scores = {}
    meanafter3_scores = {}
    meanafter4_scores = {}
    meanafter5_scores = {}
    meanafter6_scores = {}
    allpcawsing_scores = {}
    allpcawsingafter1_scores = {}
    allpcawsingafter2_scores = {}
    allpcawsingafter3_scores = {}
    allpcawsingafter4_scores = {}
    allpcawsingafter5_scores = {}
    first_max_avg_scores = {}
    second_max_avg_scores = {}
    third_max_avg_scores = {}
    fourth_max_avg_scores = {}
    fifth_max_avg_scores = {}
    sixth_max_avg_scores = {}
    seventh_max_avg_scores = {}
    p25_avg_scores = {}
    p50_avg_scores = {}
    p75_avg_scores = {}
    mean_avg_scores = {}
    IQRmean_avg_scores = {}
    meanafter1_avg_scores = {}
    meanafter2_avg_scores = {}
    meanafter3_avg_scores = {}
    meanafter4_avg_scores = {}
    meanafter5_avg_scores = {}
    meanafter6_avg_scores = {}
    allpcawsing_avg_scores = {}
    allpcawsingafter1_avg_scores = {}
    allpcawsingafter2_avg_scores = {}
    allpcawsingafter3_avg_scores = {}
    allpcawsingafter4_avg_scores = {}
    allpcawsingafter5_avg_scores = {}
    sorted_compound_scores = {}
    pca_directions = {}
    for classname in tqdm(classnames):
        ensemble_single_scores[classname],first_max_scores[classname],second_max_scores[classname],third_max_scores[classname],fourth_max_scores[classname],fifth_max_scores[classname],sixth_max_scores[classname],seventh_max_scores[classname],p25_scores[classname],p50_scores[classname],p75_scores[classname],mean_scores[classname],IQRmean_scores[classname], meanafter1_scores[classname], meanafter2_scores[classname], meanafter3_scores[classname], meanafter4_scores[classname], meanafter5_scores[classname], meanafter6_scores[classname], allpcawsing_scores[classname], allpcawsingafter1_scores[classname], allpcawsingafter2_scores[classname], allpcawsingafter3_scores[classname], allpcawsingafter4_scores[classname], allpcawsingafter5_scores[classname], sorted_compound_scores[classname], pca_directions[classname] = predict_oneclass(classname,ensemble_single_cossims,compound_cossims,classname2compoundprompts)
        first_max_avg_scores[classname] = 0.5 * first_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        second_max_avg_scores[classname] = 0.5 * second_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        third_max_avg_scores[classname] = 0.5 * third_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        fourth_max_avg_scores[classname] = 0.5 * fourth_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        fifth_max_avg_scores[classname] = 0.5 * fifth_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        sixth_max_avg_scores[classname] = 0.5 * sixth_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        seventh_max_avg_scores[classname] = 0.5 * seventh_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        p25_avg_scores[classname] = 0.5 * p25_scores[classname] + 0.5 * ensemble_single_scores[classname]
        p50_avg_scores[classname] = 0.5 * p50_scores[classname] + 0.5 * ensemble_single_scores[classname]
        p75_avg_scores[classname] = 0.5 * p75_scores[classname] + 0.5 * ensemble_single_scores[classname]
        mean_avg_scores[classname] = 0.5 * mean_scores[classname] + 0.5 * ensemble_single_scores[classname]
        IQRmean_avg_scores[classname] = 0.5 * IQRmean_scores[classname] + 0.5 * ensemble_single_scores[classname]
        meanafter1_avg_scores[classname] = 0.5 * meanafter1_scores[classname] + 0.5 * ensemble_single_scores[classname]
        meanafter2_avg_scores[classname] = 0.5 * meanafter2_scores[classname] + 0.5 * ensemble_single_scores[classname]
        meanafter3_avg_scores[classname] = 0.5 * meanafter3_scores[classname] + 0.5 * ensemble_single_scores[classname]
        meanafter4_avg_scores[classname] = 0.5 * meanafter4_scores[classname] + 0.5 * ensemble_single_scores[classname]
        meanafter5_avg_scores[classname] = 0.5 * meanafter5_scores[classname] + 0.5 * ensemble_single_scores[classname]
        meanafter6_avg_scores[classname] = 0.5 * meanafter6_scores[classname] + 0.5 * ensemble_single_scores[classname]
        allpcawsing_avg_scores[classname] = 0.5 * allpcawsing_scores[classname] + 0.5 * ensemble_single_scores[classname]
        allpcawsingafter1_avg_scores[classname] = 0.5 * allpcawsingafter1_scores[classname] + 0.5 * ensemble_single_scores[classname]
        allpcawsingafter2_avg_scores[classname] = 0.5 * allpcawsingafter2_scores[classname] + 0.5 * ensemble_single_scores[classname]
        allpcawsingafter3_avg_scores[classname] = 0.5 * allpcawsingafter3_scores[classname] + 0.5 * ensemble_single_scores[classname]
        allpcawsingafter4_avg_scores[classname] = 0.5 * allpcawsingafter4_scores[classname] + 0.5 * ensemble_single_scores[classname]
        allpcawsingafter5_avg_scores[classname] = 0.5 * allpcawsingafter5_scores[classname] + 0.5 * ensemble_single_scores[classname]

    if SAVE_EXTRA:
        extra = {'impaths' : impaths, 'gts' : gts, 'ensemble_single_scores' : ensemble_single_scores, 'first_max_scores' : first_max_scores, 'second_max_scores' : second_max_scores}
        with open(os.path.join(plot_dir, 'extra.pkl'), 'wb') as f:
            pickle.dump(extra, f)

        extra = {'impaths' : impaths, 'gts' : {classname : np.array([row[classname] for row in gts]) for classname in classnames}, 'ensemble_single_scores_calib' : ensemble_single_scores, 'sorted_compound_scores' : sorted_compound_scores, 'ensemble_single_scores_uncalib' : ensemble_single_scores_uncalib, 'allpcawsing_scores' : allpcawsing_scores, 'allpcawsing_avg_scores' : allpcawsing_avg_scores, 'pca_directions' : pca_directions}
        with open(os.path.join(plot_dir, '%s_test_%s_extra_sorted_compound_scores.pkl'%(dataset_name.split('_')[0], model_type)), 'wb') as f:
            pickle.dump(extra, f)

#    #histogram gifs
#    print('histogram gifs...')
#    make_histogram_and_histosweep_gifs(ensemble_single_scores, first_max_scores, second_max_scores, third_max_scores, gts, plot_dir, dataset_name, do_calibration)

    calib_suffix = ('_without_calibration' if not do_calibration else '')
    os.makedirs(os.path.join(os.path.dirname(plot_dir), 'result_files%s'%(calib_suffix)), exist_ok=True)
    suffix = ''
    if use_tagclip:
        suffix = '_TagCLIPlog%drowcalibbase%d'%(tagclip_use_log, tagclip_use_rowcalibbase)

    if use_taidpt:
        suffix = '_%susstrong%dthemstrong%dseed%drowcalibbase%d'%(taidpt_name, taidpt_us_strong, taidpt_them_strong, taidpt_seed, taidpt_or_comc_use_rowcalibbase)

    results_pkl_filename = os.path.join(os.path.dirname(plot_dir), 'result_files%s'%(calib_suffix), dataset_name.split('_')[0] + '_test_' + model_type + '_results%s.pkl'%(suffix))
    results_csv_filename = os.path.join(os.path.dirname(plot_dir), 'result_files%s'%(calib_suffix), dataset_name.split('_')[0] + '_test_' + model_type + '_results%s.csv'%(suffix))

    #PCA
    print('pca...')
    first_max_pca_scores = {}
    second_max_pca_scores = {}
    third_max_pca_scores = {}
    fourth_max_pca_scores = {}
    fifth_max_pca_scores = {}
    sixth_max_pca_scores = {}
    seventh_max_pca_scores = {}
    p25_pca_scores = {}
    p50_pca_scores = {}
    p75_pca_scores = {}
    mean_pca_scores = {}
    IQRmean_pca_scores = {}
    meanafter1_pca_scores = {}
    meanafter2_pca_scores = {}
    meanafter3_pca_scores = {}
    meanafter4_pca_scores = {}
    meanafter5_pca_scores = {}
    meanafter6_pca_scores = {}
    allpcawsing_pca_scores = {}
    allpcawsingafter1_pca_scores = {}
    allpcawsingafter2_pca_scores = {}
    allpcawsingafter3_pca_scores = {}
    allpcawsingafter4_pca_scores = {}
    allpcawsingafter5_pca_scores = {}
    first_max_pca_directions = {}
    second_max_pca_directions = {}
    third_max_pca_directions = {}
    sixth_max_pca_directions = {}
    mean_pca_directions = {}
    first_max_pca_explvars = {}
    second_max_pca_explvars = {}
    third_max_pca_explvars = {}
    for classname in tqdm(classnames):
        first_max_pca_scores[classname], first_max_pca_directions[classname], first_max_pca_explvars[classname] = do_PCA_oneclass(first_max_scores[classname], ensemble_single_scores[classname])
        second_max_pca_scores[classname], second_max_pca_directions[classname], second_max_pca_explvars[classname] = do_PCA_oneclass(second_max_scores[classname], ensemble_single_scores[classname])
        third_max_pca_scores[classname], third_max_pca_directions[classname], third_max_pca_explvars[classname] = do_PCA_oneclass(third_max_scores[classname], ensemble_single_scores[classname])
        fourth_max_pca_scores[classname], _, __ = do_PCA_oneclass(fourth_max_scores[classname], ensemble_single_scores[classname])
        fifth_max_pca_scores[classname], _, __ = do_PCA_oneclass(fifth_max_scores[classname], ensemble_single_scores[classname])
        sixth_max_pca_scores[classname], sixth_max_pca_directions[classname], __ = do_PCA_oneclass(sixth_max_scores[classname], ensemble_single_scores[classname])
        seventh_max_pca_scores[classname], _, __ = do_PCA_oneclass(seventh_max_scores[classname], ensemble_single_scores[classname])
        p25_pca_scores[classname], _, __ = do_PCA_oneclass(p25_scores[classname], ensemble_single_scores[classname])
        p50_pca_scores[classname], _, __ = do_PCA_oneclass(p50_scores[classname], ensemble_single_scores[classname])
        p75_pca_scores[classname], _, __ = do_PCA_oneclass(p75_scores[classname], ensemble_single_scores[classname])
        mean_pca_scores[classname], mean_pca_directions[classname], __ = do_PCA_oneclass(mean_scores[classname], ensemble_single_scores[classname])
        IQRmean_pca_scores[classname], _, __ = do_PCA_oneclass(IQRmean_scores[classname], ensemble_single_scores[classname])
        meanafter1_pca_scores[classname], _, __ = do_PCA_oneclass(meanafter1_scores[classname], ensemble_single_scores[classname])
        meanafter2_pca_scores[classname], _, __ = do_PCA_oneclass(meanafter2_scores[classname], ensemble_single_scores[classname])
        meanafter3_pca_scores[classname], _, __ = do_PCA_oneclass(meanafter3_scores[classname], ensemble_single_scores[classname])
        meanafter4_pca_scores[classname], _, __ = do_PCA_oneclass(meanafter4_scores[classname], ensemble_single_scores[classname])
        meanafter5_pca_scores[classname], _, __ = do_PCA_oneclass(meanafter5_scores[classname], ensemble_single_scores[classname])
        meanafter6_pca_scores[classname], _, __ = do_PCA_oneclass(meanafter6_scores[classname], ensemble_single_scores[classname])
        allpcawsing_pca_scores[classname], _, __ = do_PCA_oneclass(allpcawsing_scores[classname], ensemble_single_scores[classname])
        allpcawsingafter1_pca_scores[classname], _, __ = do_PCA_oneclass(allpcawsingafter1_scores[classname], ensemble_single_scores[classname])
        allpcawsingafter2_pca_scores[classname], _, __ = do_PCA_oneclass(allpcawsingafter2_scores[classname], ensemble_single_scores[classname])
        allpcawsingafter3_pca_scores[classname], _, __ = do_PCA_oneclass(allpcawsingafter3_scores[classname], ensemble_single_scores[classname])
        allpcawsingafter4_pca_scores[classname], _, __ = do_PCA_oneclass(allpcawsingafter4_scores[classname], ensemble_single_scores[classname])
        allpcawsingafter5_pca_scores[classname], _, __ = do_PCA_oneclass(allpcawsingafter5_scores[classname], ensemble_single_scores[classname])

    #compute APs
    print('APs...')
    ensemble_single_APs, first_max_APs, second_max_APs, third_max_APs, fourth_max_APs, fifth_max_APs, sixth_max_APs, seventh_max_APs, first_max_avg_APs, second_max_avg_APs, third_max_avg_APs, fourth_max_avg_APs, fifth_max_avg_APs, sixth_max_avg_APs, seventh_max_avg_APs, first_max_pca_APs, second_max_pca_APs, third_max_pca_APs, fourth_max_pca_APs, fifth_max_pca_APs, sixth_max_pca_APs, seventh_max_pca_APs = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    p25_APs, p50_APs, p75_APs, mean_APs, IQRmean_APs = {}, {}, {}, {}, {}
    p25_avg_APs, p50_avg_APs, p75_avg_APs, mean_avg_APs, IQRmean_avg_APs = {}, {}, {}, {}, {}
    p25_pca_APs, p50_pca_APs, p75_pca_APs, mean_pca_APs, IQRmean_pca_APs = {}, {}, {}, {}, {}
    meanafter1_APs, meanafter2_APs, meanafter3_APs, meanafter4_APs, meanafter5_APs, meanafter6_APs = {}, {}, {}, {}, {}, {}
    meanafter1_avg_APs, meanafter2_avg_APs, meanafter3_avg_APs, meanafter4_avg_APs, meanafter5_avg_APs, meanafter6_avg_APs = {}, {}, {}, {}, {}, {}
    meanafter1_pca_APs, meanafter2_pca_APs, meanafter3_pca_APs, meanafter4_pca_APs, meanafter5_pca_APs, meanafter6_pca_APs = {}, {}, {}, {}, {}, {}
    allpcawsing_APs, allpcawsing_avg_APs, allpcawsing_pca_APs = {}, {}, {}
    allpcawsingafter1_APs, allpcawsingafter1_avg_APs, allpcawsingafter1_pca_APs = {}, {}, {}
    allpcawsingafter2_APs, allpcawsingafter2_avg_APs, allpcawsingafter2_pca_APs = {}, {}, {}
    allpcawsingafter3_APs, allpcawsingafter3_avg_APs, allpcawsingafter3_pca_APs = {}, {}, {}
    allpcawsingafter4_APs, allpcawsingafter4_avg_APs, allpcawsingafter4_pca_APs = {}, {}, {}
    allpcawsingafter5_APs, allpcawsingafter5_avg_APs, allpcawsingafter5_pca_APs = {}, {}, {}
    for classname in tqdm(classnames):
        gts_arr = np.array([gts_row[classname] for gts_row in gts])
        ensemble_single_APs[classname] = 100.0 * average_precision(ensemble_single_scores[classname], gts_arr)
        first_max_APs[classname] = 100.0 * average_precision(first_max_scores[classname], gts_arr)
        second_max_APs[classname] = 100.0 * average_precision(second_max_scores[classname], gts_arr)
        third_max_APs[classname] = 100.0 * average_precision(third_max_scores[classname], gts_arr)
        fourth_max_APs[classname] = 100.0 * average_precision(fourth_max_scores[classname], gts_arr)
        fifth_max_APs[classname] = 100.0 * average_precision(fifth_max_scores[classname], gts_arr)
        sixth_max_APs[classname] = 100.0 * average_precision(sixth_max_scores[classname], gts_arr)
        seventh_max_APs[classname] = 100.0 * average_precision(seventh_max_scores[classname], gts_arr)
        first_max_avg_APs[classname] = 100.0 * average_precision(first_max_avg_scores[classname], gts_arr)
        second_max_avg_APs[classname] = 100.0 * average_precision(second_max_avg_scores[classname], gts_arr)
        third_max_avg_APs[classname] = 100.0 * average_precision(third_max_avg_scores[classname], gts_arr)
        fourth_max_avg_APs[classname] = 100.0 * average_precision(fourth_max_avg_scores[classname], gts_arr)
        fifth_max_avg_APs[classname] = 100.0 * average_precision(fifth_max_avg_scores[classname], gts_arr)
        sixth_max_avg_APs[classname] = 100.0 * average_precision(sixth_max_avg_scores[classname], gts_arr)
        seventh_max_avg_APs[classname] = 100.0 * average_precision(seventh_max_avg_scores[classname], gts_arr)
        first_max_pca_APs[classname] = 100.0 * average_precision(first_max_pca_scores[classname], gts_arr)
        second_max_pca_APs[classname] = 100.0 * average_precision(second_max_pca_scores[classname], gts_arr)
        third_max_pca_APs[classname] = 100.0 * average_precision(third_max_pca_scores[classname], gts_arr)
        fourth_max_pca_APs[classname] = 100.0 * average_precision(fourth_max_pca_scores[classname], gts_arr)
        fifth_max_pca_APs[classname] = 100.0 * average_precision(fifth_max_pca_scores[classname], gts_arr)
        sixth_max_pca_APs[classname] = 100.0 * average_precision(sixth_max_pca_scores[classname], gts_arr)
        seventh_max_pca_APs[classname] = 100.0 * average_precision(seventh_max_pca_scores[classname], gts_arr)
        p25_APs[classname] = 100.0 * average_precision(p25_scores[classname], gts_arr)
        p50_APs[classname] = 100.0 * average_precision(p50_scores[classname], gts_arr)
        p75_APs[classname] = 100.0 * average_precision(p75_scores[classname], gts_arr)
        mean_APs[classname] = 100.0 * average_precision(mean_scores[classname], gts_arr)
        IQRmean_APs[classname] = 100.0 * average_precision(IQRmean_scores[classname], gts_arr)
        meanafter1_APs[classname] = 100.0 * average_precision(meanafter1_scores[classname], gts_arr)
        meanafter2_APs[classname] = 100.0 * average_precision(meanafter2_scores[classname], gts_arr)
        meanafter3_APs[classname] = 100.0 * average_precision(meanafter3_scores[classname], gts_arr)
        meanafter4_APs[classname] = 100.0 * average_precision(meanafter4_scores[classname], gts_arr)
        meanafter5_APs[classname] = 100.0 * average_precision(meanafter5_scores[classname], gts_arr)
        meanafter6_APs[classname] = 100.0 * average_precision(meanafter6_scores[classname], gts_arr)
        allpcawsing_APs[classname] = 100.0 * average_precision(allpcawsing_scores[classname], gts_arr)
        allpcawsingafter1_APs[classname] = 100.0 * average_precision(allpcawsingafter1_scores[classname], gts_arr)
        allpcawsingafter2_APs[classname] = 100.0 * average_precision(allpcawsingafter2_scores[classname], gts_arr)
        allpcawsingafter3_APs[classname] = 100.0 * average_precision(allpcawsingafter3_scores[classname], gts_arr)
        allpcawsingafter4_APs[classname] = 100.0 * average_precision(allpcawsingafter4_scores[classname], gts_arr)
        allpcawsingafter5_APs[classname] = 100.0 * average_precision(allpcawsingafter5_scores[classname], gts_arr)
        p25_avg_APs[classname] = 100.0 * average_precision(p25_avg_scores[classname], gts_arr)
        p50_avg_APs[classname] = 100.0 * average_precision(p50_avg_scores[classname], gts_arr)
        p75_avg_APs[classname] = 100.0 * average_precision(p75_avg_scores[classname], gts_arr)
        mean_avg_APs[classname] = 100.0 * average_precision(mean_avg_scores[classname], gts_arr)
        IQRmean_avg_APs[classname] = 100.0 * average_precision(IQRmean_avg_scores[classname], gts_arr)
        meanafter1_avg_APs[classname] = 100.0 * average_precision(meanafter1_avg_scores[classname], gts_arr)
        meanafter2_avg_APs[classname] = 100.0 * average_precision(meanafter2_avg_scores[classname], gts_arr)
        meanafter3_avg_APs[classname] = 100.0 * average_precision(meanafter3_avg_scores[classname], gts_arr)
        meanafter4_avg_APs[classname] = 100.0 * average_precision(meanafter4_avg_scores[classname], gts_arr)
        meanafter5_avg_APs[classname] = 100.0 * average_precision(meanafter5_avg_scores[classname], gts_arr)
        meanafter6_avg_APs[classname] = 100.0 * average_precision(meanafter6_avg_scores[classname], gts_arr)
        allpcawsing_avg_APs[classname] = 100.0 * average_precision(allpcawsing_avg_scores[classname], gts_arr)
        allpcawsingafter1_avg_APs[classname] = 100.0 * average_precision(allpcawsingafter1_avg_scores[classname], gts_arr)
        allpcawsingafter2_avg_APs[classname] = 100.0 * average_precision(allpcawsingafter2_avg_scores[classname], gts_arr)
        allpcawsingafter3_avg_APs[classname] = 100.0 * average_precision(allpcawsingafter3_avg_scores[classname], gts_arr)
        allpcawsingafter4_avg_APs[classname] = 100.0 * average_precision(allpcawsingafter4_avg_scores[classname], gts_arr)
        allpcawsingafter5_avg_APs[classname] = 100.0 * average_precision(allpcawsingafter5_avg_scores[classname], gts_arr)
        p25_pca_APs[classname] = 100.0 * average_precision(p25_pca_scores[classname], gts_arr)
        p50_pca_APs[classname] = 100.0 * average_precision(p50_pca_scores[classname], gts_arr)
        p75_pca_APs[classname] = 100.0 * average_precision(p75_pca_scores[classname], gts_arr)
        mean_pca_APs[classname] = 100.0 * average_precision(mean_pca_scores[classname], gts_arr)
        IQRmean_pca_APs[classname] = 100.0 * average_precision(IQRmean_pca_scores[classname], gts_arr)
        meanafter1_pca_APs[classname] = 100.0 * average_precision(meanafter1_pca_scores[classname], gts_arr)
        meanafter2_pca_APs[classname] = 100.0 * average_precision(meanafter2_pca_scores[classname], gts_arr)
        meanafter3_pca_APs[classname] = 100.0 * average_precision(meanafter3_pca_scores[classname], gts_arr)
        meanafter4_pca_APs[classname] = 100.0 * average_precision(meanafter4_pca_scores[classname], gts_arr)
        meanafter5_pca_APs[classname] = 100.0 * average_precision(meanafter5_pca_scores[classname], gts_arr)
        meanafter6_pca_APs[classname] = 100.0 * average_precision(meanafter6_pca_scores[classname], gts_arr)
        allpcawsing_pca_APs[classname] = 100.0 * average_precision(allpcawsing_pca_scores[classname], gts_arr)
        allpcawsingafter1_pca_APs[classname] = 100.0 * average_precision(allpcawsingafter1_pca_scores[classname], gts_arr)
        allpcawsingafter2_pca_APs[classname] = 100.0 * average_precision(allpcawsingafter2_pca_scores[classname], gts_arr)
        allpcawsingafter3_pca_APs[classname] = 100.0 * average_precision(allpcawsingafter3_pca_scores[classname], gts_arr)
        allpcawsingafter4_pca_APs[classname] = 100.0 * average_precision(allpcawsingafter4_pca_scores[classname], gts_arr)
        allpcawsingafter5_pca_APs[classname] = 100.0 * average_precision(allpcawsingafter5_pca_scores[classname], gts_arr)

    #compute mAP
    ensemble_single_mAP = np.mean([ensemble_single_APs[classname] for classname in classnames])
    first_max_mAP = np.mean([first_max_APs[classname] for classname in classnames])
    second_max_mAP = np.mean([second_max_APs[classname] for classname in classnames])
    third_max_mAP = np.mean([third_max_APs[classname] for classname in classnames])
    fourth_max_mAP = np.mean([fourth_max_APs[classname] for classname in classnames])
    fifth_max_mAP = np.mean([fifth_max_APs[classname] for classname in classnames])
    sixth_max_mAP = np.mean([sixth_max_APs[classname] for classname in classnames])
    seventh_max_mAP = np.mean([seventh_max_APs[classname] for classname in classnames])
    first_max_avg_mAP = np.mean([first_max_avg_APs[classname] for classname in classnames])
    second_max_avg_mAP = np.mean([second_max_avg_APs[classname] for classname in classnames])
    third_max_avg_mAP = np.mean([third_max_avg_APs[classname] for classname in classnames])
    fourth_max_avg_mAP = np.mean([fourth_max_avg_APs[classname] for classname in classnames])
    fifth_max_avg_mAP = np.mean([fifth_max_avg_APs[classname] for classname in classnames])
    sixth_max_avg_mAP = np.mean([sixth_max_avg_APs[classname] for classname in classnames])
    seventh_max_avg_mAP = np.mean([seventh_max_avg_APs[classname] for classname in classnames])
    first_max_pca_mAP = np.mean([first_max_pca_APs[classname] for classname in classnames])
    second_max_pca_mAP = np.mean([second_max_pca_APs[classname] for classname in classnames])
    third_max_pca_mAP = np.mean([third_max_pca_APs[classname] for classname in classnames])
    fourth_max_pca_mAP = np.mean([fourth_max_pca_APs[classname] for classname in classnames])
    fifth_max_pca_mAP = np.mean([fifth_max_pca_APs[classname] for classname in classnames])
    sixth_max_pca_mAP = np.mean([sixth_max_pca_APs[classname] for classname in classnames])
    seventh_max_pca_mAP = np.mean([seventh_max_pca_APs[classname] for classname in classnames])
    p25_mAP = np.mean([p25_APs[classname] for classname in classnames])
    p50_mAP = np.mean([p50_APs[classname] for classname in classnames])
    p75_mAP = np.mean([p75_APs[classname] for classname in classnames])
    mean_mAP = np.mean([mean_APs[classname] for classname in classnames])
    IQRmean_mAP = np.mean([IQRmean_APs[classname] for classname in classnames])
    meanafter1_mAP = np.mean([meanafter1_APs[classname] for classname in classnames])
    meanafter2_mAP = np.mean([meanafter2_APs[classname] for classname in classnames])
    meanafter3_mAP = np.mean([meanafter3_APs[classname] for classname in classnames])
    meanafter4_mAP = np.mean([meanafter4_APs[classname] for classname in classnames])
    meanafter5_mAP = np.mean([meanafter5_APs[classname] for classname in classnames])
    meanafter6_mAP = np.mean([meanafter6_APs[classname] for classname in classnames])
    allpcawsing_mAP = np.mean([allpcawsing_APs[classname] for classname in classnames])
    allpcawsingafter1_mAP = np.mean([allpcawsingafter1_APs[classname] for classname in classnames])
    allpcawsingafter2_mAP = np.mean([allpcawsingafter2_APs[classname] for classname in classnames])
    allpcawsingafter3_mAP = np.mean([allpcawsingafter3_APs[classname] for classname in classnames])
    allpcawsingafter4_mAP = np.mean([allpcawsingafter4_APs[classname] for classname in classnames])
    allpcawsingafter5_mAP = np.mean([allpcawsingafter5_APs[classname] for classname in classnames])
    p25_avg_mAP = np.mean([p25_avg_APs[classname] for classname in classnames])
    p50_avg_mAP = np.mean([p50_avg_APs[classname] for classname in classnames])
    p75_avg_mAP = np.mean([p75_avg_APs[classname] for classname in classnames])
    mean_avg_mAP = np.mean([mean_avg_APs[classname] for classname in classnames])
    IQRmean_avg_mAP = np.mean([IQRmean_avg_APs[classname] for classname in classnames])
    meanafter1_avg_mAP = np.mean([meanafter1_avg_APs[classname] for classname in classnames])
    meanafter2_avg_mAP = np.mean([meanafter2_avg_APs[classname] for classname in classnames])
    meanafter3_avg_mAP = np.mean([meanafter3_avg_APs[classname] for classname in classnames])
    meanafter4_avg_mAP = np.mean([meanafter4_avg_APs[classname] for classname in classnames])
    meanafter5_avg_mAP = np.mean([meanafter5_avg_APs[classname] for classname in classnames])
    meanafter6_avg_mAP = np.mean([meanafter6_avg_APs[classname] for classname in classnames])
    allpcawsing_avg_mAP = np.mean([allpcawsing_avg_APs[classname] for classname in classnames])
    allpcawsingafter1_avg_mAP = np.mean([allpcawsingafter1_avg_APs[classname] for classname in classnames])
    allpcawsingafter2_avg_mAP = np.mean([allpcawsingafter2_avg_APs[classname] for classname in classnames])
    allpcawsingafter3_avg_mAP = np.mean([allpcawsingafter3_avg_APs[classname] for classname in classnames])
    allpcawsingafter4_avg_mAP = np.mean([allpcawsingafter4_avg_APs[classname] for classname in classnames])
    allpcawsingafter5_avg_mAP = np.mean([allpcawsingafter5_avg_APs[classname] for classname in classnames])
    p25_pca_mAP = np.mean([p25_pca_APs[classname] for classname in classnames])
    p50_pca_mAP = np.mean([p50_pca_APs[classname] for classname in classnames])
    p75_pca_mAP = np.mean([p75_pca_APs[classname] for classname in classnames])
    mean_pca_mAP = np.mean([mean_pca_APs[classname] for classname in classnames])
    IQRmean_pca_mAP = np.mean([IQRmean_pca_APs[classname] for classname in classnames])
    meanafter1_pca_mAP = np.mean([meanafter1_pca_APs[classname] for classname in classnames])
    meanafter2_pca_mAP = np.mean([meanafter2_pca_APs[classname] for classname in classnames])
    meanafter3_pca_mAP = np.mean([meanafter3_pca_APs[classname] for classname in classnames])
    meanafter4_pca_mAP = np.mean([meanafter4_pca_APs[classname] for classname in classnames])
    meanafter5_pca_mAP = np.mean([meanafter5_pca_APs[classname] for classname in classnames])
    meanafter6_pca_mAP = np.mean([meanafter6_pca_APs[classname] for classname in classnames])
    allpcawsing_pca_mAP = np.mean([allpcawsing_pca_APs[classname] for classname in classnames])
    allpcawsingafter1_pca_mAP = np.mean([allpcawsingafter1_pca_APs[classname] for classname in classnames])
    allpcawsingafter2_pca_mAP = np.mean([allpcawsingafter2_pca_APs[classname] for classname in classnames])
    allpcawsingafter3_pca_mAP = np.mean([allpcawsingafter3_pca_APs[classname] for classname in classnames])
    allpcawsingafter4_pca_mAP = np.mean([allpcawsingafter4_pca_APs[classname] for classname in classnames])
    allpcawsingafter5_pca_mAP = np.mean([allpcawsingafter5_pca_APs[classname] for classname in classnames])

    #plot APs
    print('plot APs...')
    plot_APs(ensemble_single_APs, first_max_APs, second_max_APs, third_max_APs, first_max_avg_APs, second_max_avg_APs, third_max_avg_APs, first_max_pca_APs, second_max_pca_APs, third_max_pca_APs, plot_dir, dataset_name, model_type, do_calibration)

    #PCA explvars
    print('pca explvars plots...')
    make_pca_explvars_plot(first_max_avg_APs, first_max_pca_APs, first_max_pca_explvars, '1stmax', '1stmax_avg', plot_dir, do_calibration, model_type, dataset_name)
    make_pca_explvars_plot(second_max_avg_APs, second_max_pca_APs, second_max_pca_explvars, '2ndmax', '2ndmax_avg', plot_dir, do_calibration, model_type, dataset_name)
    make_pca_explvars_plot(third_max_avg_APs, third_max_pca_APs, third_max_pca_explvars, '3rdmax', '3rdmax_avg', plot_dir, do_calibration, model_type, dataset_name)
    make_pca_explvars_plot(ensemble_single_APs, first_max_pca_APs, first_max_pca_explvars, '1stmax', 'ensemble80', plot_dir, do_calibration, model_type, dataset_name)
    make_pca_explvars_plot(ensemble_single_APs, second_max_pca_APs, second_max_pca_explvars, '2ndmax', 'ensemble80', plot_dir, do_calibration, model_type, dataset_name)
    make_pca_explvars_plot(ensemble_single_APs, third_max_pca_APs, third_max_pca_explvars, '3rdmax', 'ensemble80', plot_dir, do_calibration, model_type, dataset_name)

    #compute blending curves, including mAP curves
    print('blending curves...')
    first_max_curves, second_max_curves, third_max_curves, sixth_max_curves, mean_curves = {}, {}, {}, {}, {}
    for classname in tqdm(classnames):
        first_max_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, first_max_scores, gts)
        second_max_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, second_max_scores, gts)
        third_max_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, third_max_scores, gts)
        sixth_max_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, sixth_max_scores, gts)
        mean_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, mean_scores, gts)

    first_max_mAP_curve = np.mean([first_max_curves[classname] for classname in classnames], axis=0)
    second_max_mAP_curve = np.mean([second_max_curves[classname] for classname in classnames], axis=0)
    third_max_mAP_curve = np.mean([third_max_curves[classname] for classname in classnames], axis=0)
    sixth_max_mAP_curve = np.mean([sixth_max_curves[classname] for classname in classnames], axis=0)
    mean_mAP_curve = np.mean([mean_curves[classname] for classname in classnames], axis=0)
    first_max_blendoracle_mAP = np.mean([np.amax(first_max_curves[classname]) for classname in classnames])
    second_max_blendoracle_mAP = np.mean([np.amax(second_max_curves[classname]) for classname in classnames])
    third_max_blendoracle_mAP = np.mean([np.amax(third_max_curves[classname]) for classname in classnames])
    print('blending Xs: %s'%(str(BLENDING_XS)))
    print('blending first max mAPs: %s (peak_x=%f, peak_mAP=%f)'%(str(first_max_mAP_curve), BLENDING_XS[np.argmax(first_max_mAP_curve)], np.amax(first_max_mAP_curve)))
    print('blending second max mAPs: %s (peak_x=%f, peak_mAP=%f)'%(str(second_max_mAP_curve), BLENDING_XS[np.argmax(second_max_mAP_curve)], np.amax(second_max_mAP_curve)))
    print('blending third max mAPs: %s (peak_x=%f, peak_mAP=%f)'%(str(third_max_mAP_curve), BLENDING_XS[np.argmax(third_max_mAP_curve)], np.amax(third_max_mAP_curve)))
    print('blendoracle first max mAP: %s'%(str(first_max_blendoracle_mAP)))
    print('blendoracle second max mAP: %s'%(str(second_max_blendoracle_mAP)))
    print('blendoracle third max mAP: %s'%(str(third_max_blendoracle_mAP)))

    selectoracle_APs = {}
    for classname in classnames:
        selectoracle_APs[classname] = max([first_max_avg_APs[classname], first_max_pca_APs[classname], second_max_avg_APs[classname], second_max_pca_APs[classname], third_max_avg_APs[classname], third_max_pca_APs[classname], ensemble_single_APs[classname]])

    selectoracle_mAP = np.mean([selectoracle_APs[classname] for classname in classnames])

    f = open(results_csv_filename, 'w')
    f.write('method,mAP\n')
    results = {}
    print('RESULTS...')
    baseline_name = 'ensemble_single'
    if use_tagclip:
        baseline_name = 'TagCLIP'

    if use_taidpt:
        baseline_name = taidpt_name

    append_to_results('%s_uncalibrated'%(baseline_name), ensemble_single_uncalibrated_mAP, ensemble_single_uncalibrated_APs, results, f)
    append_to_results('%s_calibrated'%(baseline_name), ensemble_single_mAP, ensemble_single_APs, results, f)
    append_to_results('first_max', first_max_mAP, first_max_APs, results, f)
    append_to_results('second_max', second_max_mAP, second_max_APs, results, f)
    append_to_results('third_max', third_max_mAP, third_max_APs, results, f)
    append_to_results('fourth_max', fourth_max_mAP, fourth_max_APs, results, f)
    append_to_results('fifth_max', fifth_max_mAP, fifth_max_APs, results, f)
    append_to_results('sixth_max', sixth_max_mAP, sixth_max_APs, results, f)
    append_to_results('seventh_max', seventh_max_mAP, seventh_max_APs, results, f)
    append_to_results('p25compounds', p25_mAP, p25_APs, results, f)
    append_to_results('p50compounds', p50_mAP, p50_APs, results, f)
    append_to_results('p75compounds', p75_mAP, p75_APs, results, f)
    append_to_results('meancompounds', mean_mAP, mean_APs, results, f)
    append_to_results('meanafter1compounds', meanafter1_mAP, meanafter1_APs, results, f)
    append_to_results('meanafter2compounds', meanafter2_mAP, meanafter2_APs, results, f)
    append_to_results('meanafter3compounds', meanafter3_mAP, meanafter3_APs, results, f)
    append_to_results('meanafter4compounds', meanafter4_mAP, meanafter4_APs, results, f)
    append_to_results('meanafter5compounds', meanafter5_mAP, meanafter5_APs, results, f)
    append_to_results('meanafter6compounds', meanafter6_mAP, meanafter6_APs, results, f)
    append_to_results('allpcawsing', allpcawsing_mAP, allpcawsing_APs, results, f)
    append_to_results('allpcawsingafter1', allpcawsingafter1_mAP, allpcawsingafter1_APs, results, f)
    append_to_results('allpcawsingafter2', allpcawsingafter2_mAP, allpcawsingafter2_APs, results, f)
    append_to_results('allpcawsingafter3', allpcawsingafter3_mAP, allpcawsingafter3_APs, results, f)
    append_to_results('allpcawsingafter4', allpcawsingafter4_mAP, allpcawsingafter4_APs, results, f)
    append_to_results('allpcawsingafter5', allpcawsingafter5_mAP, allpcawsingafter5_APs, results, f)
    append_to_results('IQRmeancompounds', IQRmean_mAP, IQRmean_APs, results, f)
    append_to_results('first_max_avg', first_max_avg_mAP, first_max_avg_APs, results, f)
    append_to_results('second_max_avg', second_max_avg_mAP, second_max_avg_APs, results, f)
    append_to_results('third_max_avg', third_max_avg_mAP, third_max_avg_APs, results, f)
    append_to_results('fourth_max_avg', fourth_max_avg_mAP, fourth_max_avg_APs, results, f)
    append_to_results('fifth_max_avg', fifth_max_avg_mAP, fifth_max_avg_APs, results, f)
    append_to_results('sixth_max_avg', sixth_max_avg_mAP, sixth_max_avg_APs, results, f)
    append_to_results('seventh_max_avg', seventh_max_avg_mAP, seventh_max_avg_APs, results, f)
    append_to_results('p25compounds_avg', p25_avg_mAP, p25_avg_APs, results, f)
    append_to_results('p50compounds_avg', p50_avg_mAP, p50_avg_APs, results, f)
    append_to_results('p75compounds_avg', p75_avg_mAP, p75_avg_APs, results, f)
    append_to_results('meancompounds_avg', mean_avg_mAP, mean_avg_APs, results, f)
    append_to_results('IQRmeancompounds_avg', IQRmean_avg_mAP, IQRmean_avg_APs, results, f)
    append_to_results('meanafter1compounds_avg', meanafter1_avg_mAP, meanafter1_avg_APs, results, f)
    append_to_results('meanafter2compounds_avg', meanafter2_avg_mAP, meanafter2_avg_APs, results, f)
    append_to_results('meanafter3compounds_avg', meanafter3_avg_mAP, meanafter3_avg_APs, results, f)
    append_to_results('meanafter4compounds_avg', meanafter4_avg_mAP, meanafter4_avg_APs, results, f)
    append_to_results('meanafter5compounds_avg', meanafter5_avg_mAP, meanafter5_avg_APs, results, f)
    append_to_results('meanafter6compounds_avg', meanafter6_avg_mAP, meanafter6_avg_APs, results, f)
    append_to_results('allpcawsing_avg', allpcawsing_avg_mAP, allpcawsing_avg_APs, results, f)
    append_to_results('allpcawsingafter1_avg', allpcawsingafter1_avg_mAP, allpcawsingafter1_avg_APs, results, f)
    append_to_results('allpcawsingafter2_avg', allpcawsingafter2_avg_mAP, allpcawsingafter2_avg_APs, results, f)
    append_to_results('allpcawsingafter3_avg', allpcawsingafter3_avg_mAP, allpcawsingafter3_avg_APs, results, f)
    append_to_results('allpcawsingafter4_avg', allpcawsingafter4_avg_mAP, allpcawsingafter4_avg_APs, results, f)
    append_to_results('allpcawsingafter5_avg', allpcawsingafter5_avg_mAP, allpcawsingafter5_avg_APs, results, f)
    append_to_results('first_max_pca', first_max_pca_mAP, first_max_pca_APs, results, f)
    append_to_results('second_max_pca', second_max_pca_mAP, second_max_pca_APs, results, f)
    append_to_results('third_max_pca', third_max_pca_mAP, third_max_pca_APs, results, f)
    append_to_results('fourth_max_pca', fourth_max_pca_mAP, fourth_max_pca_APs, results, f)
    append_to_results('fifth_max_pca', fifth_max_pca_mAP, fifth_max_pca_APs, results, f)
    append_to_results('sixth_max_pca', sixth_max_pca_mAP, sixth_max_pca_APs, results, f)
    append_to_results('seventh_max_pca', seventh_max_pca_mAP, seventh_max_pca_APs, results, f)
    append_to_results('p25compounds_pca', p25_pca_mAP, p25_pca_APs, results, f)
    append_to_results('p50compounds_pca', p50_pca_mAP, p50_pca_APs, results, f)
    append_to_results('p75compounds_pca', p75_pca_mAP, p75_pca_APs, results, f)
    append_to_results('meancompounds_pca', mean_pca_mAP, mean_pca_APs, results, f)
    append_to_results('IQRmeancompounds_pca', IQRmean_pca_mAP, IQRmean_pca_APs, results, f)
    append_to_results('meanafter1compounds_pca', meanafter1_pca_mAP, meanafter1_pca_APs, results, f)
    append_to_results('meanafter2compounds_pca', meanafter2_pca_mAP, meanafter2_pca_APs, results, f)
    append_to_results('meanafter3compounds_pca', meanafter3_pca_mAP, meanafter3_pca_APs, results, f)
    append_to_results('meanafter4compounds_pca', meanafter4_pca_mAP, meanafter4_pca_APs, results, f)
    append_to_results('meanafter5compounds_pca', meanafter5_pca_mAP, meanafter5_pca_APs, results, f)
    append_to_results('meanafter6compounds_pca', meanafter6_pca_mAP, meanafter6_pca_APs, results, f)
    append_to_results('allpcawsing_pca', allpcawsing_pca_mAP, allpcawsing_pca_APs, results, f)
    append_to_results('allpcawsingafter1_pca', allpcawsingafter1_pca_mAP, allpcawsingafter1_pca_APs, results, f)
    append_to_results('allpcawsingafter2_pca', allpcawsingafter2_pca_mAP, allpcawsingafter2_pca_APs, results, f)
    append_to_results('allpcawsingafter3_pca', allpcawsingafter3_pca_mAP, allpcawsingafter3_pca_APs, results, f)
    append_to_results('allpcawsingafter4_pca', allpcawsingafter4_pca_mAP, allpcawsingafter4_pca_APs, results, f)
    append_to_results('allpcawsingafter5_pca', allpcawsingafter5_pca_mAP, allpcawsingafter5_pca_APs, results, f)
    append_to_results('first_max_constblendoracle', np.amax(first_max_mAP_curve), {c : first_max_curves[c][np.argmax(first_max_mAP_curve)] for c in classnames}, results, f)
    append_to_results('second_max_constblendoracle', np.amax(second_max_mAP_curve), {c : second_max_curves[c][np.argmax(second_max_mAP_curve)] for c in classnames}, results, f)
    append_to_results('third_max_constblendoracle', np.amax(third_max_mAP_curve), {c : third_max_curves[c][np.argmax(third_max_mAP_curve)] for c in classnames}, results, f)
    append_to_results('first_max_blendoracle', first_max_blendoracle_mAP, {c : np.amax(first_max_curves[c]) for c in classnames}, results, f)
    append_to_results('second_max_blendoracle', second_max_blendoracle_mAP, {c : np.amax(second_max_curves[c]) for c in classnames}, results, f)
    append_to_results('third_max_blendoracle', third_max_blendoracle_mAP, {c : np.amax(third_max_curves[c]) for c in classnames}, results, f)
    append_to_results('selectoracle', selectoracle_mAP, selectoracle_APs, results, f)
    f.close()
    with open(results_pkl_filename, 'wb') as f:
        pickle.dump(results, f)

#    #compute probability that other class in compound prompt is in image
#    print('topk presences...')
#    first_max_presences_given_neg, second_max_presences_given_neg, third_max_presences_given_neg, first_max_presences_given_pos, second_max_presences_given_pos, third_max_presences_given_pos = {}, {}, {}, {}, {}, {}
#    for classname in tqdm(classnames):
#        first_max_presences_given_neg[classname] = compute_topk_presence(classname,compound_cossims,gts,classname2compoundprompts,compoundprompt2classnames,1,0)
#        second_max_presences_given_neg[classname] = compute_topk_presence(classname,compound_cossims,gts,classname2compoundprompts,compoundprompt2classnames,2,0)
#        third_max_presences_given_neg[classname] = compute_topk_presence(classname,compound_cossims,gts,classname2compoundprompts,compoundprompt2classnames,3,0)
#        first_max_presences_given_pos[classname] = compute_topk_presence(classname,compound_cossims,gts,classname2compoundprompts,compoundprompt2classnames,1,1)
#        second_max_presences_given_pos[classname] = compute_topk_presence(classname,compound_cossims,gts,classname2compoundprompts,compoundprompt2classnames,2,1)
#        third_max_presences_given_pos[classname] = compute_topk_presence(classname,compound_cossims,gts,classname2compoundprompts,compoundprompt2classnames,3,1)

#    #compute mean cossim as we go down the rank
#    print('topk mean cossims...')
#    first_max_mean_cossims_given_neg, second_max_mean_cossims_given_neg, third_max_mean_cossims_given_neg, first_max_mean_cossims_given_pos, second_max_mean_cossims_given_pos, third_max_mean_cossims_given_pos, first_max_mean_cossims_given_all, second_max_mean_cossims_given_all, third_max_mean_cossims_given_all = {}, {}, {}, {}, {}, {}, {}, {}, {}
 #   for classname in tqdm(classnames):
 #       first_max_mean_cossims_given_neg[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,1,0)
 #       second_max_mean_cossims_given_neg[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,2,0)
 #       third_max_mean_cossims_given_neg[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,3,0)
 #       first_max_mean_cossims_given_pos[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,1,1)
 #       second_max_mean_cossims_given_pos[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,2,1)
 #       third_max_mean_cossims_given_pos[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,3,1)
 #       first_max_mean_cossims_given_all[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,1,None)
 #       second_max_mean_cossims_given_all[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,2,None)
 #       third_max_mean_cossims_given_all[classname] = compute_topk_mean_cossim(classname,compound_cossims,gts,classname2compoundprompts,3,None)

    #plot blending curves
    print('plot blending curves...')
#    plot_blending_curves(first_max_curves, second_max_curves, third_max_curves, first_max_mAP_curve, second_max_mAP_curve, third_max_mAP_curve, plot_dir, dataset_name, model_type, do_calibration)
    plot_blending_curves_helper([sixth_max_curves, mean_curves], [sixth_max_mAP_curve, mean_mAP_curve], [sixth_max_pca_directions, mean_pca_directions], ['sixth_max', 'mean'], plot_dir, dataset_name, model_type, do_calibration)

#    #plot topk presences
#    print('plot topk presences...')
#    plot_topk_presences(first_max_presences_given_neg, second_max_presences_given_neg, third_max_presences_given_neg, first_max_presences_given_pos, second_max_presences_given_pos, third_max_presences_given_pos, plot_dir, dataset_name, do_calibration)

 #   #plot topk mean cossims
#    print('plot topk mean cossims...')
#    plot_topk_mean_cossims(first_max_mean_cossims_given_neg, second_max_mean_cossims_given_neg, third_max_mean_cossims_given_neg, first_max_mean_cossims_given_pos, second_max_mean_cossims_given_pos, third_max_mean_cossims_given_pos, first_max_mean_cossims_given_all, second_max_mean_cossims_given_all, third_max_mean_cossims_given_all, plot_dir, dataset_name, do_calibration)
#    plot_topk_mean_cossim_features_vs_perfdiff(first_max_mean_cossims_given_all, second_max_mean_cossims_given_all, third_max_mean_cossims_given_all, ensemble_single_APs, third_max_pca_APs, plot_dir, dataset_name, do_calibration)



def usage():
    print('Usage: python second_max_experiments.py <dataset_name> <do_calibration>')


if __name__ == '__main__':
    second_max_experiments(*(sys.argv[1:]))
