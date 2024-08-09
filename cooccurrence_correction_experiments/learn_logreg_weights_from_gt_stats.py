import os
import sys
import copy
import math
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from harvest_training_gts import get_data_manager
from compute_mAP import average_precision
from logistic_regression_experiment import OUT_PARENT_DIR


#INPUT_FILENAME = os.path.join(OUT_PARENT_DIR, 'logistic_regression_%s_logits_standardize0_balance0_C0.1.pkl')
#OUTPUT_FILENAME = os.path.join(OUT_PARENT_DIR, '../learn_logreg_weights_from_gt_stats/learn_logreg_weights_from_gt_stats_%s_logits_standardize0_balance0_C0.1.pkl')
#HEATMAP_FILENAME = os.path.join(OUT_PARENT_DIR, '../learn_logreg_weights_from_gt_stats/learn_logreg_weights_from_gt_stats_%s_logits_standardize0_balance0_C0.1_heatmaps.png')
##INPUT_FILENAME = os.path.join(OUT_PARENT_DIR, 'logistic_regression_%s_cossims_standardize0_balance0_C1000.0.pkl')
##OUTPUT_FILENAME = os.path.join(OUT_PARENT_DIR, '../learn_logreg_weights_from_gt_stats/learn_logreg_weights_from_gt_stats_%s_cossims_standardize0_balance0_C1000.0.pkl')
##HEATMAP_FILENAME = os.path.join(OUT_PARENT_DIR, '../learn_logreg_weights_from_gt_stats/learn_logreg_weights_from_gt_stats_%s_cossims_standardize0_balance0_C1000.0_heatmaps.png')
NUM_PLOTS_PER_ROW = 4
FIGWIDTH, FIGHEIGHT = 9, 9


VMIN_VMAX_DICT = {'logreg_weights' : (None, None), \
                    'pred_W' : (None, None), \
                    'Prij' : (0, None), \
                    'marg_prod' : (0, None), \
                    'Prij_over_marg_prod' : (0, 2), \
                    'Pri_cond_j' : (0, None), \
                    'Prj_cond_i' : (0, None)}


#returns input_filename, probs_input_filename, output_filename
def get_filenames(dataset_name, input_type, C, params, miniclass):
    p = params
    if miniclass:
        probs_input_filename = None
    else:
        probs_input_filename = os.path.join(OUT_PARENT_DIR, 'logistic_regression_%s_probs_standardize0_balance0_C0.1.pkl'%(dataset_name.split('_')[0]))

    input_filename = os.path.join(OUT_PARENT_DIR, 'logistic_regression_%s_%s_standardize0_balance0_C%s_miniclass%d.pkl'%(dataset_name.split('_')[0], input_type, str(C), miniclass))
    output_filename = os.path.join(OUT_PARENT_DIR, '../learn_logreg_weights_from_gt_stats/learn_logreg_weights_from_gt_stats_%s_%s_standardize0_balance0_C%s_miniclass%d_XTXpinv%d_appendinputstats%d_isolatediags%d.pkl'%(dataset_name.split('_')[0], input_type, str(C), miniclass, p['matmul_by_XTXpinv'], p['append_input_stats'], p['isolate_diags']))
    heatmap_filename = os.path.join(OUT_PARENT_DIR, '../learn_logreg_weights_from_gt_stats/learn_logreg_weights_from_gt_stats_%s_%s_standardize0_balance0_C%s_miniclass%d_XTXpinv%d_appendinputstats%d_isolatediags%d_heatmaps.png'%(dataset_name.split('_')[0], input_type, str(C), miniclass, p['matmul_by_XTXpinv'], p['append_input_stats'], p['isolate_diags']))

    return input_filename, probs_input_filename, output_filename, heatmap_filename


#return as matrix
def get_logreg_weights(logreg_dict):
    my_clf = logreg_dict['model']
    W = []
    for one_clf in my_clf:
        assert(len(one_clf.named_steps) == 1)
        W.append(np.squeeze(one_clf.named_steps['clf'].coef_))

    return np.array(W)


def no_diag_ify(A):
    A = copy.deepcopy(A)
    np.fill_diagonal(A, 0)
    return A


def matmul_by_XTXpinv(gt_stats, logreg_dict, exclude_list):
    X_train = logreg_dict['data']['X_train']
    XTX_pinv = np.linalg.pinv(X_train.T @ X_train)
    my_keys = sorted(gt_stats.keys())
    for k in my_keys:
        if k in exclude_list:
            continue

        assert('XTXpinv_matmul_' + k not in gt_stats)
        gt_stats['XTXpinv_matmul_' + k] = XTX_pinv @ gt_stats[k]

    return gt_stats


def append_input_stats(gt_stats, logreg_dict, logreg_dict_probs, miniclass):
    if miniclass:
        assert(logreg_dict_probs is None)
    else:
        assert(logreg_dict_probs is not None)

    X_train = logreg_dict['data']['X_train']
    XTX = X_train.T @ X_train
    XTX_pinv = np.linalg.pinv(XTX)
    if miniclass:
        gt_stats['XTX'] = XTX
        gt_stats['XTX_pinv'] = XTX_pinv
        return gt_stats

    Xprob_train = logreg_dict_probs['data']['X_train']
    pseudo_margs = np.mean(Xprob_train, axis=0)
    pseudo_Pri = np.tile(pseudo_margs[:,np.newaxis], (1, len(pseudo_margs)))
    pseudo_Prj = np.tile(pseudo_margs[np.newaxis,:], (len(pseudo_margs), 1))
    pseudo_marg_prod = pseudo_Pri * pseudo_Prj
    pseudo_Prij = Xprob_train.T @ Xprob_train / (1.0 * Xprob_train.shape[0])
    pseudo_Pri_cond_j = pseudo_Prij / pseudo_Prj
    pseudo_Prj_cond_i = pseudo_Prij / pseudo_Pri
    pseudo_Prij_over_marg_prod = pseudo_Prij / pseudo_marg_prod
    new_gt_stats = {'XTX' : XTX, 'XTX_pinv' : XTX_pinv, 'pseudo_Pri' : pseudo_Pri, 'pseudo_Prj' : pseudo_Prj, 'pseudo_marg_prod' : pseudo_marg_prod, 'pseudo_Prij' : pseudo_Prij, 'pseudo_Pri_cond_j' : pseudo_Pri_cond_j, 'pseudo_Prj_cond_i' : pseudo_Prj_cond_i, 'pseudo_Prij_over_marg_prod' : pseudo_Prij_over_marg_prod}
    for k in sorted(new_gt_stats.keys()):
        assert(k not in gt_stats)
        gt_stats[k] = new_gt_stats[k]

    return gt_stats


def isolate_diags(gt_stats, exclude_list):
    my_keys = sorted(gt_stats.keys())
    for k in my_keys:
        if k in exclude_list:
            continue

        assert(k + '_nodiag' not in gt_stats)
        gt_stats[k + '_nodiag'] = no_diag_ify(gt_stats[k])

    return gt_stats


#return dict mapping to matrices
#stats will include: Pr(i), Pr(j), Pr(i)*Pr(j), Pr(i,j), Pr(i|j), Pr(j|i), Pr(i,j)/(Pr(i)*Pr(j)), Identity
def get_gt_stats(logreg_dict, logreg_dict_probs, params, miniclass):
    p = params
    y_train = logreg_dict['data']['y_train']
    gt_margs = np.mean(y_train, axis=0)
    Pri = np.tile(gt_margs[:,np.newaxis], (1, len(gt_margs)))
    Prj = np.tile(gt_margs[np.newaxis,:], (len(gt_margs), 1))
    marg_prod = Pri * Prj
    Prij = y_train.T @ y_train / (1.0 * y_train.shape[0])
    Pri_cond_j = Prij / Prj
    Prj_cond_i = Prij / Pri
    Prij_over_marg_prod = Prij / marg_prod
    gt_stats = {'Identity' : np.eye(len(gt_margs)), 'Pri' : Pri, 'Prj' : Prj, 'marg_prod' : marg_prod, 'Prij' : Prij, 'Pri_cond_j' : Pri_cond_j, 'Prj_cond_i' : Prj_cond_i, 'Prij_over_marg_prod' : Prij_over_marg_prod}
    if p['matmul_by_XTXpinv']:
        gt_stats = matmul_by_XTXpinv(gt_stats, logreg_dict, ['Identity'])

    if p['append_input_stats']:
        gt_stats = append_input_stats(gt_stats, logreg_dict, logreg_dict_probs, miniclass)

    if p['isolate_diags']:
        gt_stats = isolate_diags(gt_stats, ['Identity', 'Pri', 'Prj', 'XTXpinv_matmul_Pri', 'XTXpinv_matmul_Prj', 'pseudo_Pri', 'pseudo_Prj'])

    return gt_stats


#return dict with entries "bias" and "coefs", the latter of which will map to a dict
def do_learning(gt_stats, logreg_weights):
    stats_keys = sorted(gt_stats.keys())
    y = logreg_weights.flatten()[:,np.newaxis]
    X = []
    X.append(np.ones(len(logreg_weights.flatten())))
    for k in stats_keys:
        X.append(gt_stats[k].flatten())

    X = np.array(X)
    X = np.ascontiguousarray(X.T)
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    w = np.squeeze(w)
    bias = w[0]
    coefs = {k : ww for k, ww in zip(stats_keys, w[1:])}
    return {'bias' : bias, 'coefs' : coefs}


#will look at R2 of reconstructed vs actual weights
#and also look at train_mAP and test_mAP produced by reconstructed weights
def evaluate(gt_stats, learned_mapping, logreg_weights, logreg_dict):
    pred_W = sum([learned_mapping['coefs'][k] * gt_stats[k] for k in sorted(gt_stats.keys())])
    pred_W += learned_mapping['bias']
    pred_W = np.reshape(pred_W, logreg_weights.shape)
    R2 = np.corrcoef(logreg_weights.flatten(), pred_W.flatten())[0,1] ** 2
    X_train, y_train, X_test, y_test = logreg_dict['data']['X_train'], logreg_dict['data']['y_train'], logreg_dict['data']['X_test'], logreg_dict['data']['y_test']
    train_preds = (pred_W @ X_train.T).T
    test_preds = (pred_W @ X_test.T).T
    train_class_APs = np.array([100.0 * average_precision(train_preds[:,i], y_train[:,i]) for i in range(y_train.shape[1])])
    train_mAP = np.mean(train_class_APs)
    test_class_APs = np.array([100.0 * average_precision(test_preds[:,i], y_test[:,i]) for i in range(y_test.shape[1])])
    test_mAP = np.mean(test_class_APs)

    #using actual logreg weights
    logreg_train_preds = (logreg_weights @ X_train.T).T
    logreg_train_class_APs = np.array([100.0 * average_precision(logreg_train_preds[:,i], y_train[:,i]) for i in range(y_train.shape[1])])
    logreg_train_mAP = np.mean(logreg_train_class_APs)
    logreg_test_preds = (logreg_weights @ X_test.T).T
    logreg_test_class_APs = np.array([100.0 * average_precision(logreg_test_preds[:,i], y_test[:,i]) for i in range(y_test.shape[1])])
    logreg_test_mAP = np.mean(logreg_test_class_APs)

    result = {'pred_W' : pred_W, 'R2' : R2, 'train_preds' : train_preds, 'test_preds' : test_preds, 'train_class_APs' : train_class_APs, 'test_class_APs' : test_class_APs, 'train_mAP' : train_mAP, 'test_mAP' : test_mAP, 'logreg_train_mAP' : logreg_train_mAP, 'logreg_test_mAP' : logreg_test_mAP}
    return result


def learn_logreg_weights_from_gt_stats(dataset_name, input_type, C, miniclass, matmul_by_XTXpinv, append_input_stats, isolate_diags):
    C = float(C)
    miniclass = int(miniclass)
    matmul_by_XTXpinv = int(matmul_by_XTXpinv)
    append_input_stats = int(append_input_stats)
    isolate_diags = int(isolate_diags)
    if miniclass:
        assert(input_type == 'cossims')

    p = {'matmul_by_XTXpinv' : matmul_by_XTXpinv, 'append_input_stats' : append_input_stats, 'isolate_diags' : isolate_diags}
    input_filename, probs_input_filename, output_filename, heatmap_filename = get_filenames(dataset_name, input_type, C, p, miniclass)

    print('load data...')
    with open(input_filename, 'rb') as f:
        logreg_dict = pickle.load(f)

    if miniclass:
        assert(probs_input_filename is None)
        logreg_dict_probs = None
    else:
        with open(probs_input_filename, 'rb') as f:
            logreg_dict_probs = pickle.load(f)

    print('get logreg weights...')
    logreg_weights = get_logreg_weights(logreg_dict)
    print('get gt stats...')
    gt_stats = get_gt_stats(logreg_dict, logreg_dict_probs, p, miniclass)
    print('learn...')
    learned_mapping = do_learning(gt_stats, logreg_weights)
    print('evaluate...')
    eval_dict = evaluate(gt_stats, learned_mapping, logreg_weights, logreg_dict)
    print(learned_mapping)
    print('dataset_name=%s, input_type=%s, C=%s, miniclass=%d, params=%s, R2=%f, train_mAP=%f, test_mAP=%f, logreg_train_mAP=%f, logreg_test_mAP=%f'%(dataset_name.split('_')[0], input_type, str(C), miniclass, str(p), eval_dict['R2'], eval_dict['train_mAP'], eval_dict['test_mAP'], eval_dict['logreg_train_mAP'], eval_dict['logreg_test_mAP']))
    output_dict = {'logreg_weights' : logreg_weights, 'gt_stats' : gt_stats, 'learned_mapping' : learned_mapping, 'eval' : eval_dict, 'params' : p, 'input_type' : input_type, 'C' : C, 'miniclass' : miniclass, 'input_filename' : input_filename, 'dataset_name' : dataset_name.split('_')[0]}
    with open(output_filename, 'wb') as f:
        pickle.dump(output_dict, f)

    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    classnames = [classnames[i] for i in logreg_dict['data']['class_indices']]
    make_heatmap(logreg_weights, gt_stats, eval_dict, logreg_dict['eval'], classnames, dataset_name, C, heatmap_filename)


def get_nondiagonal_min_and_max(A):
    A = copy.deepcopy(A)
    np.fill_diagonal(A, np.nan)
    return np.nanmin(A), np.nanmax(A)


def render_one_heatmap(A, my_title, classnames, vmin, vmax, fig, ax):
    other_vmin, other_vmax = get_nondiagonal_min_and_max(A)
    if vmin is None:
        vmin = other_vmin

    if vmax is None:
        vmax = other_vmax

    hmap = ax.imshow(A, vmin=vmin, vmax=vmax, aspect='auto')
    fig.colorbar(hmap, ax=ax)
    ax.set_xticks(ticks=np.arange(len(classnames)), labels=classnames, rotation=90)
    ax.set_yticks(ticks=np.arange(len(classnames)), labels=classnames)
    ax.set_xlabel('j')
    ax.set_ylabel('i')
    ax.set_title(my_title)
    return vmin, vmax


def make_heatmap(logreg_weights, gt_stats, eval_dict, logreg_eval_dict, classnames, dataset_name, C, heatmap_filename):
    plottable_keys = [k for k in sorted(gt_stats.keys()) if '_nodiag' not in k and 'Identity' not in k]
    total_num_plots = len(plottable_keys) + 2
    num_rows = int(math.ceil(total_num_plots / NUM_PLOTS_PER_ROW))
    plt.clf()
    fig, axs = plt.subplots(num_rows, NUM_PLOTS_PER_ROW, figsize=(NUM_PLOTS_PER_ROW * FIGWIDTH, 2 * FIGHEIGHT))
    wvmin, wvmax = render_one_heatmap(logreg_weights, 'W', classnames, VMIN_VMAX_DICT['logreg_weights'][0], VMIN_VMAX_DICT['logreg_weights'][1], fig, axs[0][0])
    render_one_heatmap(eval_dict['pred_W'], 'reconstructed W', classnames, wvmin, wvmax, fig, axs[0][1])
    for t, k in enumerate(plottable_keys):
        i = (t+2) // NUM_PLOTS_PER_ROW
        j = (t+2) % NUM_PLOTS_PER_ROW
        if k in VMIN_VMAX_DICT:
            vmin, vmax = VMIN_VMAX_DICT[k]
        else:
            vmin, vmax = None, None

        render_one_heatmap(gt_stats[k], k, classnames, vmin, vmax, fig, axs[i][j])

    fig.suptitle('%s: C=%s, train_mAP=%.3f, input_train_mAP=%.3f, logreg_train_mAP=%.3f, test_mAP=%.3f, input_test_mAP=%.3f, logreg_test_mAP=%.3f'%(dataset_name.split('_')[0], str(C), eval_dict['train_mAP'], logreg_eval_dict['input_train_mAP'], eval_dict['logreg_train_mAP'], eval_dict['test_mAP'], logreg_eval_dict['input_test_mAP'], eval_dict['logreg_test_mAP']))

    plt.tight_layout()
    plt.savefig(heatmap_filename)
    plt.clf()


def usage():
    print('Usage: python learn_logreg_weights_from_gt_stats.py <dataset_name> <input_type> <C> <miniclass> <matmul_by_XTXpinv> <append_input_stats> <isolate_diags>')


if __name__ == '__main__':
    learn_logreg_weights_from_gt_stats(*(sys.argv[1:]))
