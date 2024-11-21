import os
import sys
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

#COCO only
COCO_ENSEMBLE_SINGLE_FILENAME = os.path.join(BASE_DIR, 'COCO2014/result_table_input_train_noaug_COCO2014_ViT-L14336px_ensemble_80.pkl')
COCO_COMPOUND_FILENAME = os.path.join(BASE_DIR, 'COCO2014/clip_comprenhensive_prompt_600_named.csv')
COCO_VENKATESH_GTS_FILENAME = os.path.join(BASE_DIR, 'COCO2014/Ground_Truth_Dataset_with_Class_Names.csv')
COCO_VENKATESH_CLASSNAMES_RENAMER = {'food bowl' : 'bowl'}
COCO_KEVIN_CLASSNAMES_RENAMER = {'motor bike' : 'motorcycle', 'aeroplane' : 'airplane'}

#nuswide only
NUSWIDE_ENSEMBLE_SINGLE_FILENAME = os.path.join(BASE_DIR, 'nuswide/result_table_input_train_noaug_nuswide_ViT-L14336px_ensemble_80.pkl')
NUSWIDE_SIMPLE_SINGLE_FILENAME = os.path.join(BASE_DIR, 'nuswide/compound_cossims_nuswide_ViT-L14336px_only_use_for_simple_single.pkl')
NUSWIDE_COMPOUND_FILENAME = os.path.join(BASE_DIR, 'nuswide/compound_cossims_nuswide_ViT-L14336px.pkl')

BLENDING_XS = np.arange(21) / 20

GIF_CLASSNAMES_DICT = {'nuswide_partial' : ['airport', 'animal', 'coral', 'reflection', 'swimmers', 'sun', 'fish'], 'COCO2014_partial' : ['airplane', 'baseball glove', 'book', 'cup', 'mouse', 'spoon', 'zebra']}
BLENDING_XS_FOR_GIF = np.arange(201) / 200
NUM_BINS_NEG = 200
NUM_BINS_POS = 80
FPS = 20


def depluralize(s):
    if s[-1] == 's':
        return s[:-1]
    else:
        return s


def stringmatch_to_compoundprompt(classnames, compoundprompt):
    for classname in classnames:
        assert(re.sub(r'[^a-zA-Z]+', ' ', classname.lower()).strip() == classname)
        assert('  ' not in classname)

    sorted_classnames = sorted(classnames, key = lambda c: len(c), reverse=True)
    matches = []
    substrate = ' ' + re.sub(r'[^a-zA-Z]+', ' ', compoundprompt.lower()) + ' '
    for classname in sorted_classnames:
        flag = False
        for c in sorted(set([classname, depluralize(classname), classname + 's', classname + 'es', classname.replace('person', 'people').replace('child', 'children').replace('man', 'men').replace('foot', 'feet').replace('goose', 'geese').replace('mouse', 'mice').replace('die', 'dice').replace('tooth', 'teeth').replace('louse', 'lice').replace('leaf', 'leaves').replace('wolf', 'wolves').replace('knife', 'knives').replace('cactus', 'cacti').replace('shelf', 'shelves').replace('calf', 'calves')])):
            if ' ' + c + ' ' in substrate:
                flag = True
                substrate = substrate.replace(' ' + c + ' ', ' ! ')

        if flag:
            matches.append(classname)

    return matches


def do_padding(classnames, classname2compoundprompts, compoundprompt2classnames, simple_single_cossims, compound_cossims):
    for c in classnames:
        if c not in classname2compoundprompts:
            classname2compoundprompts[c] = []

        if len(classname2compoundprompts[c]) < 3:
            print('insufficient compound prompts, will pad with simple single: %s'%(str((c, classname2compoundprompts[c]))))
            num_pads = 3 - len(classname2compoundprompts[c])
            for _ in range(num_pads):
                classname2compoundprompts[c].append(c)

            assert(c not in compoundprompt2classnames)
            compoundprompt2classnames[c] = [c]
            for i in range(len(compound_cossims)):
                compound_cossims[i][c] = simple_single_cossims[i][c]
                assert(c in compound_cossims[i])

    return classname2compoundprompts, compoundprompt2classnames, compound_cossims


def load_data_COCO():

    #load Kevin's file
    with open(COCO_ENSEMBLE_SINGLE_FILENAME, 'rb') as f:
        d_ensemble_single = pickle.load(f)

    #load Venkatesh's file (both cossims and gts)
    df_gts = pd.read_csv(COCO_VENKATESH_GTS_FILENAME)
    df_cossims = pd.read_csv(COCO_COMPOUND_FILENAME)

    #make sure Kevin's gts match Venkatesh's gts
    assert(np.all(d_ensemble_single['gts'] == df_gts.to_numpy()))

    #split Venkatesh's cossims file into single and compound parts
    num_classes = len(d_ensemble_single['classnames'])
    df_single_cossims, df_compound_cossims = df_cossims.iloc[:, :num_classes], df_cossims.iloc[:, num_classes:]

    #fix Kevin's classnames from "motor bike" ==> "motorcycle" and "aeroplane" ==> "airplane"
    d_ensemble_single['classnames'] = [(COCO_KEVIN_CLASSNAMES_RENAMER[s] if s in COCO_KEVIN_CLASSNAMES_RENAMER else s) for s in d_ensemble_single['classnames']]

    #fix Venkatesh's single classnames from "food bowl" ==> "bowl"
    df_single_cossims.rename(columns=COCO_VENKATESH_CLASSNAMES_RENAMER, inplace=True)

    #now make sure that Venkatesh single, Venkatesh gts, and Kevin single classnames all match, including ordering
    assert(df_single_cossims.columns.tolist() == df_gts.columns.tolist())
    assert(d_ensemble_single['classnames'] == df_gts.columns.tolist())

    #crop Kevin's file to the first 15k rows
    num_rows = df_single_cossims.shape[0]
    df_gts = df_gts.iloc[:num_rows, :]
    d_ensemble_single['cossims'], d_ensemble_single['gts'] = d_ensemble_single['cossims'][:num_rows, :], d_ensemble_single['gts'][:num_rows, :]

    #do string-matching stuff
    classname2compoundprompts = {}
    compoundprompt2classnames = {}
    for compoundprompt in tqdm(df_compound_cossims.columns.tolist()):
        matches = stringmatch_to_compoundprompt(d_ensemble_single['classnames'], compoundprompt)
        compoundprompt2classnames[compoundprompt] = matches
        for classname in matches:
            if classname not in classname2compoundprompts:
                classname2compoundprompts[classname] = []

            classname2compoundprompts[classname].append(compoundprompt)

    #make cossims and gts structures
    compound_cossims = df_compound_cossims.to_dict(orient='records')
    simple_single_cossims = df_single_cossims.to_dict(orient='records')
    gts = df_gts.to_dict(orient='records')
    ensemble_single_cossims = []
    for cossims_row in d_ensemble_single['cossims']:
        ensemble_single_cossims.append({k : cossim for k,cossim in zip(d_ensemble_single['classnames'],cossims_row)})

    #padding
    classname2compoundprompts, compoundprompt2classnames, compound_cossims = do_padding(d_ensemble_single['classnames'], classname2compoundprompts, compoundprompt2classnames, simple_single_cossims, compound_cossims)

    #return everything
    return gts, ensemble_single_cossims, simple_single_cossims, compound_cossims, classname2compoundprompts, compoundprompt2classnames


def load_data_nuswide():

    #load pkl files
    with open(NUSWIDE_ENSEMBLE_SINGLE_FILENAME, 'rb') as f:
        d_ensemble_single = pickle.load(f)

    with open(NUSWIDE_SIMPLE_SINGLE_FILENAME, 'rb') as f:
        d_simple_single = pickle.load(f)

    with open(NUSWIDE_COMPOUND_FILENAME, 'rb') as f:
        d_compound = pickle.load(f)

    #get classnames, gts, compoundprompts, double-check everything
    classnames = d_ensemble_single['classnames']
    num_classes = len(classnames)
    assert(d_simple_single['compoundprompts'][:num_classes] == classnames)
    gts_arr = d_ensemble_single['gts']
    assert(np.all(d_simple_single['gts'] == gts_arr))
    assert(np.all(d_compound['gts'] == gts_arr))
    compoundprompts = d_compound['compoundprompts']
    assert(all([' ' in p for p in compoundprompts]))

    #do string-matching stuff
    classname2compoundprompts = {}
    compoundprompt2classnames = {}
    for compoundprompt in tqdm(compoundprompts):
        matches = stringmatch_to_compoundprompt(classnames, compoundprompt)
        compoundprompt2classnames[compoundprompt] = matches
        for classname in matches:
            if classname not in classname2compoundprompts:
                classname2compoundprompts[classname] = []

            classname2compoundprompts[classname].append(compoundprompt)

    #get cossims, make structures
    ensemble_single_cossims_arr = d_ensemble_single['cossims']
    simple_single_cossims_arr = d_simple_single['cossims'][:,:num_classes]
    compound_cossims_arr = d_compound['cossims']
    gts = [{c : v for c, v in zip(classnames, gts_row)} for gts_row in gts_arr]
    ensemble_single_cossims = [{c : v for c, v in zip(classnames, cossims_row)} for cossims_row in ensemble_single_cossims_arr]
    simple_single_cossims = [{c : v for c, v in zip(classnames, cossims_row)} for cossims_row in simple_single_cossims_arr]
    compound_cossims = [{p : v for p, v in zip(compoundprompts, cossims_row)} for cossims_row in compound_cossims_arr]

    #padding
    classname2compoundprompts, compoundprompt2classnames, compound_cossims = do_padding(classnames, classname2compoundprompts, compoundprompt2classnames, simple_single_cossims, compound_cossims)

    #return everything
    return gts, ensemble_single_cossims, simple_single_cossims, compound_cossims, classname2compoundprompts, compoundprompt2classnames


#return:
#-gts (list of dicts, each mapping classname to a gt)
#-ensemble_single_cossims (ditto, but map to a cossim)
#-simple_single_cossims (ditto)
#-compound_cossims (ditto, but map prompt to a cossim)
#-classname2compoundprompts (dict mapping classname to list of compound prompts that use that classname)
#-compoundprompt2classnames (dict mapping compound prompt to list of classnames used by that prompt)
def load_data(dataset_name):
    assert(dataset_name in ['COCO2014_partial', 'nuswide_partial'])
    if dataset_name == 'COCO2014_partial':
        return load_data_COCO()
    elif dataset_name == 'nuswide_partial':
        return load_data_nuswide()
    else:
        assert(False)


def calibrate_by_row(ensemble_single_cossims, simple_single_cossims, compound_cossims):
    ensemble_single_cossims_calib = []
    compound_cossims_calib = []
    for ensemble_single_row, simple_single_row, compound_row in zip(ensemble_single_cossims, simple_single_cossims, compound_cossims):
        ensemble_mean = np.mean([ensemble_single_row[k] for k in sorted(ensemble_single_row.keys())])
        ensemble_sd = np.std([ensemble_single_row[k] for k in sorted(ensemble_single_row.keys())], ddof=1)
        simple_mean = np.mean([simple_single_row[k] for k in sorted(simple_single_row.keys())])
        simple_sd = np.std([simple_single_row[k] for k in sorted(simple_single_row.keys())], ddof=1)
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
def calibrate_data(ensemble_single_cossims, simple_single_cossims, compound_cossims):
    ensemble_single_cossims, compound_cossims = calibrate_by_row(ensemble_single_cossims, simple_single_cossims, compound_cossims)
    ensemble_single_cossims = calibrate_by_column(ensemble_single_cossims)
    compound_cossims = calibrate_by_column(compound_cossims)
    return ensemble_single_cossims, compound_cossims


#return ensemble_single_scores, first_max_scores, second_max_scores, third_max_scores as 1D np arrays
#yes, this does the whole dataset, but just for one class
#no, this does NOT do calibration, it is your responsibility to do that beforehand if you want it!
def predict_oneclass(classname,ensemble_single_cossims,compound_cossims,classname2compoundprompts):
    assert(len(ensemble_single_cossims) == len(compound_cossims))
    ensemble_single_scores = []
    first_max_scores = []
    second_max_scores = []
    third_max_scores = []
    assert(len(classname2compoundprompts[classname]) >= 3)
    for ensemble_single_row, compound_row in zip(ensemble_single_cossims, compound_cossims):
        ensemble_single_score = ensemble_single_row[classname]
        compound_scores = [compound_row[prompt] for prompt in classname2compoundprompts[classname]]
        third_max_score, second_max_score, first_max_score = np.sort(compound_scores)[-3:]
        assert(first_max_score >= second_max_score)
        assert(second_max_score >= third_max_score)
        ensemble_single_scores.append(ensemble_single_score)
        first_max_scores.append(first_max_score)
        second_max_scores.append(second_max_score)
        third_max_scores.append(third_max_score)

    return np.array(ensemble_single_scores), np.array(first_max_scores), np.array(second_max_scores), np.array(third_max_scores)


def do_PCA_oneclass(*scores_list):
    scores_arr = np.array(list(scores_list)).T
    assert(scores_arr.shape[0] > scores_arr.shape[1])
    my_pca = PCA()
    my_pca.fit(scores_arr)
    direction = my_pca.components_[0,:]
    assert(np.all(direction > 0) or np.all(direction < 0))
    if not np.all(direction > 0):
        direction = -1 * direction

    assert(np.all(direction > 0))
    direction = direction / np.sum(direction)
    pca_scores = np.squeeze(scores_arr @ direction[:,np.newaxis]) #no need to center, it's just a constant offset
    pca_direction = direction
    pca_explvars = my_pca.explained_variance_
    return pca_scores, pca_direction, pca_explvars


def compute_topk_presence(classname, compound_cossims, gts, classname2compoundprompts, compoundprompt2classnames, topk, gt_filter):
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


def plot_APs(ensemble_single_APs, first_max_APs, second_max_APs, third_max_APs, first_max_avg_APs, second_max_avg_APs, third_max_avg_APs, first_max_pca_APs, second_max_pca_APs, third_max_pca_APs, plot_dir, dataset_name, do_calibration):
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
    plot_filename = os.path.join(plot_dir, 'APs_%s_calib%d.png'%(dataset_name.split('_')[0], do_calibration))
    plt.savefig(plot_filename)
    plt.clf()


def plot_blending_curves(first_max_curves, second_max_curves, third_max_curves, first_max_mAP_curve, second_max_mAP_curve, third_max_mAP_curve, plot_dir, dataset_name, do_calibration):
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
    plt.savefig(os.path.join(plot_dir, 'blending_mAPs_%s_calib%d.png'%(dataset_name.split('_')[0], do_calibration)))
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
        plt.savefig(os.path.join(plot_dir, 'blending_per_class', '%s_blending_mAPs_%s_calib%d.png'%(classname.replace(' ', ''), dataset_name.split('_')[0], do_calibration)))
        plt.clf()


def plot_topk_presences(first_max_presences_given_neg, second_max_presences_given_neg, third_max_presences_given_neg, first_max_presences_given_pos, second_max_presences_given_pos, third_max_presences_given_pos, plot_dir, dataset_name, do_calibration):
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


def make_pca_explvars_plot(baseline_APs, topk_pca_APs, topk_pca_explvars, topk_name, baseline_name, plot_dir, do_calibration, dataset_name):
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
    plt.savefig(os.path.join(plot_dir, '%s_pca_explvars_vs_perfdiff_against_%s_%s_calib%d.png'%(topk_name, baseline_name, dataset_name.split('_')[0], do_calibration)))
    plt.clf()


def hist_fn(scores, num_bins):
    my_hist, bin_edges = np.histogram(scores, bins=num_bins, density=True)
    assert(len(bin_edges) == len(my_hist) + 1)
    return my_hist, 0.5 * (bin_edges[:-1] + bin_edges[1:]), bin_edges


#blend_x should KEVIN
def make_one_histosweep(ensemble_single_scores, topk_scores, blend_x, gts, classname, topk_name, frame_t, plot_dir, dataset_name, do_calibration):
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
    classnames = GIF_CLASSNAMES_DICT[dataset_name]
    for classname in tqdm(classnames):
        for topk_scores, topk_name in zip([first_max_scores, second_max_scores, third_max_scores], ['1stmax', '2ndmax', '3rdmax']):
            blend_x = make_histogram_gif(ensemble_single_scores[classname], topk_scores[classname], np.array([gts_row[classname] for gts_row in gts]), classname, topk_name, plot_dir, dataset_name, do_calibration)
            make_histosweep_gif(ensemble_single_scores[classname], topk_scores[classname], blend_x, np.array([gts_row[classname] for gts_row in gts]), classname, topk_name, plot_dir, dataset_name, do_calibration)


def second_max_experiments(dataset_name, do_calibration):
    do_calibration = int(do_calibration)

    print('load data...')
    gts, ensemble_single_cossims, simple_single_cossims, compound_cossims, classname2compoundprompts, compoundprompt2classnames = load_data(dataset_name)
    classnames = sorted(gts[0].keys())

    #do calibration if needed
    if do_calibration:
        print('calibration...')
        ensemble_single_cossims, compound_cossims = calibrate_data(ensemble_single_cossims, simple_single_cossims, compound_cossims)

    #get the scores, each as dict mapping from classname to 1D array
    #also get the uniform averaging scores
    print('scores...')
    ensemble_single_scores = {}
    first_max_scores = {}
    second_max_scores = {}
    third_max_scores = {}
    first_max_avg_scores = {}
    second_max_avg_scores = {}
    third_max_avg_scores = {}
    for classname in tqdm(classnames):
        ensemble_single_scores[classname],first_max_scores[classname],second_max_scores[classname],third_max_scores[classname] = predict_oneclass(classname,ensemble_single_cossims,compound_cossims,classname2compoundprompts)
        first_max_avg_scores[classname] = 0.5 * first_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        second_max_avg_scores[classname] = 0.5 * second_max_scores[classname] + 0.5 * ensemble_single_scores[classname]
        third_max_avg_scores[classname] = 0.5 * third_max_scores[classname] + 0.5 * ensemble_single_scores[classname]

    plot_dir = os.path.join(BASE_DIR, dataset_name.split('_')[0], {False : 'without_calibration', True : 'with_calibration'}[do_calibration])

#    #histogram gifs
#    print('histogram gifs...')
#    make_histogram_and_histosweep_gifs(ensemble_single_scores, first_max_scores, second_max_scores, third_max_scores, gts, plot_dir, dataset_name, do_calibration)

    #PCA
    print('pca...')
    first_max_pca_scores = {}
    second_max_pca_scores = {}
    third_max_pca_scores = {}
    first_max_pca_directions = {}
    second_max_pca_directions = {}
    third_max_pca_directions = {}
    first_max_pca_explvars = {}
    second_max_pca_explvars = {}
    third_max_pca_explvars = {}
    for classname in tqdm(classnames):
        first_max_pca_scores[classname], first_max_pca_directions[classname], first_max_pca_explvars[classname] = do_PCA_oneclass(first_max_scores[classname], ensemble_single_scores[classname])
        second_max_pca_scores[classname], second_max_pca_directions[classname], second_max_pca_explvars[classname] = do_PCA_oneclass(second_max_scores[classname], ensemble_single_scores[classname])
        third_max_pca_scores[classname], third_max_pca_directions[classname], third_max_pca_explvars[classname] = do_PCA_oneclass(third_max_scores[classname], ensemble_single_scores[classname])

    #compute APs
    print('APs...')
    ensemble_single_APs, first_max_APs, second_max_APs, third_max_APs, first_max_avg_APs, second_max_avg_APs, third_max_avg_APs, first_max_pca_APs, second_max_pca_APs, third_max_pca_APs = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for classname in tqdm(classnames):
        gts_arr = np.array([gts_row[classname] for gts_row in gts])
        ensemble_single_APs[classname] = 100.0 * average_precision(ensemble_single_scores[classname], gts_arr)
        first_max_APs[classname] = 100.0 * average_precision(first_max_scores[classname], gts_arr)
        second_max_APs[classname] = 100.0 * average_precision(second_max_scores[classname], gts_arr)
        third_max_APs[classname] = 100.0 * average_precision(third_max_scores[classname], gts_arr)
        first_max_avg_APs[classname] = 100.0 * average_precision(first_max_avg_scores[classname], gts_arr)
        second_max_avg_APs[classname] = 100.0 * average_precision(second_max_avg_scores[classname], gts_arr)
        third_max_avg_APs[classname] = 100.0 * average_precision(third_max_avg_scores[classname], gts_arr)
        first_max_pca_APs[classname] = 100.0 * average_precision(first_max_pca_scores[classname], gts_arr)
        second_max_pca_APs[classname] = 100.0 * average_precision(second_max_pca_scores[classname], gts_arr)
        third_max_pca_APs[classname] = 100.0 * average_precision(third_max_pca_scores[classname], gts_arr)

    #PCA explvars
    print('pca explvars plots...')
    make_pca_explvars_plot(first_max_avg_APs, first_max_pca_APs, first_max_pca_explvars, '1stmax', '1stmax_avg', plot_dir, do_calibration, dataset_name)
    make_pca_explvars_plot(second_max_avg_APs, second_max_pca_APs, second_max_pca_explvars, '2ndmax', '2ndmax_avg', plot_dir, do_calibration, dataset_name)
    make_pca_explvars_plot(third_max_avg_APs, third_max_pca_APs, third_max_pca_explvars, '3rdmax', '3rdmax_avg', plot_dir, do_calibration, dataset_name)
    make_pca_explvars_plot(ensemble_single_APs, first_max_pca_APs, first_max_pca_explvars, '1stmax', 'ensemble80', plot_dir, do_calibration, dataset_name)
    make_pca_explvars_plot(ensemble_single_APs, second_max_pca_APs, second_max_pca_explvars, '2ndmax', 'ensemble80', plot_dir, do_calibration, dataset_name)
    make_pca_explvars_plot(ensemble_single_APs, third_max_pca_APs, third_max_pca_explvars, '3rdmax', 'ensemble80', plot_dir, do_calibration, dataset_name)

    #compute blending curves, including mAP curves
    print('blending curves...')
    first_max_curves, second_max_curves, third_max_curves = {}, {}, {}
    for classname in tqdm(classnames):
        first_max_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, first_max_scores, gts)
        second_max_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, second_max_scores, gts)
        third_max_curves[classname] = make_topk_blending_curve(classname, ensemble_single_scores, third_max_scores, gts)

    first_max_mAP_curve = np.mean([first_max_curves[classname] for classname in classnames], axis=0)
    second_max_mAP_curve = np.mean([second_max_curves[classname] for classname in classnames], axis=0)
    third_max_mAP_curve = np.mean([third_max_curves[classname] for classname in classnames], axis=0)
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

    #plot APs
    print('plot APs...')
    plot_APs(ensemble_single_APs, first_max_APs, second_max_APs, third_max_APs, first_max_avg_APs, second_max_avg_APs, third_max_avg_APs, first_max_pca_APs, second_max_pca_APs, third_max_pca_APs, plot_dir, dataset_name, do_calibration)

#    #plot blending curves
#    print('plot blending curves...')
#    plot_blending_curves(first_max_curves, second_max_curves, third_max_curves, first_max_mAP_curve, second_max_mAP_curve, third_max_mAP_curve, plot_dir, dataset_name, do_calibration)

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
