import os
import sys
import copy
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from harvest_training_gts import get_data_manager
from compute_mAP import average_precision
from logistic_regression_experiment import load_data as _load_data


ALPHA = 0.05
K = 72 #75
GAMMA_LO = 0.6
GAMMA_HI = 1.4
BETA = 0.95
MAX_TOPN = 5
C_DICT = {'cossims' : 1000.0, 'logits' : 1000.0} #{'cossims' : 1000.0, 'logits' : 0.1}
OUT_PARENT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/graph_strategy')


#gt_conditionals[i,j] = Pr(j | i)


#get candidates for "loner" classes based on conditional probabilities
#this doesn't do the part where we look at the number of labels
#returns a mask
def get_loner_candidates(gt_conditionals, alpha, K):
    return (np.sum(gt_conditionals < alpha, axis=1) >= K)


#returns loner_classes, labels
#loner_classes will be a mask, labels will be binary with zeros in non-loner columns
def get_loner_classes_and_labels(clip_scores, gt_conditionals, gt_margs, alpha, K, gamma_lo, gamma_hi):
    loner_classes = get_loner_candidates(gt_conditionals, alpha, K)
    is_argmax = (clip_scores >= np.amax(clip_scores, axis=1, keepdims=True))
    pred_margs = np.mean(is_argmax, axis=0)
    good_margs = ((pred_margs >= gamma_lo * gt_margs) & (pred_margs <= gamma_hi * gt_margs))
    loner_classes = (loner_classes & good_margs)
    labels = (is_argmax & loner_classes[np.newaxis, :])
    return loner_classes, labels


#returns sorted_eligible_loners (indices)
def get_sorted_eligible_loner_classes(target, gt_conditionals, loner_classes, beta):
    conds_for_target = gt_conditionals[:,target]
    eligible_loners = np.nonzero(loner_classes & (conds_for_target >= beta))[0]
    assert(len(eligible_loners.shape) == 1)
    if eligible_loners.shape[0] == 0:
        return []

    pairs = [(i, conds_for_target[i]) for i in eligible_loners]
    sorted_pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    sorted_eligible_loners = [pair[0] for pair in sorted_pairs]
    assert(conds_for_target[sorted_eligible_loners[0]] >= conds_for_target[sorted_eligible_loners[-1]])
    return sorted_eligible_loners


#returns is_sufficient, labels_one_target
#do NOT zero out labels_one_target if insufficient!!!
def get_neighbor_classes_and_labels_one_target_one_topN(target, clip_scores, gt_conditionals, gt_margs, loner_classes, labels, labels_one_target, beta, topN, gamma_lo, gamma_hi):
    #sort eligible loner classes by conditional probability
    sorted_eligible_loners = get_sorted_eligible_loner_classes(target, gt_conditionals, loner_classes, beta)

    #get the topN score mask for each example
    assert(topN >= 2)
    is_in_topN = (clip_scores[:,target] >= np.sort(clip_scores, axis=1)[:,-topN])

    #now do bag-filling
    for eligible_loner in sorted_eligible_loners:
        new_labels = (labels[:,eligible_loner] & is_in_topN)
        labels_one_target = (labels_one_target | new_labels)
        pred_marg = np.mean(labels_one_target)
        if pred_marg >= gamma_lo * gt_margs[target] and pred_marg <= gamma_hi * gt_margs[target]:
            return True, labels_one_target

    return False, labels_one_target


#target should be an index
#returns is_neighbor (bool), labels_one_target (1D binary array)
def get_neighbor_classes_and_labels_one_target(target, clip_scores, gt_conditionals, gt_margs, loner_classes, labels, beta, max_topN, gamma_lo, gamma_hi):
    labels_one_target = np.zeros_like(labels[:,target])
    for topN in range(2, max_topN+1):
        is_sufficient, labels_one_target = get_neighbor_classes_and_labels_one_target_one_topN(target, clip_scores, gt_conditionals, gt_margs, loner_classes, labels, labels_one_target, beta, topN, gamma_lo, gamma_hi)
        if is_sufficient:
            return True, labels_one_target

    return False, np.zeros_like(labels_one_target)


def get_neighbor_classes_and_labels(clip_scores, gt_conditionals, gt_margs, loner_classes, labels, beta, max_topN, gamma_lo, gamma_hi):
    neighbor_classes = np.zeros_like(loner_classes)
    for target in tqdm(range(len(gt_margs))):
        if loner_classes[target]:
            continue

        is_neighbor, labels_one_target = get_neighbor_classes_and_labels_one_target(target, clip_scores, gt_conditionals, gt_margs, loner_classes, labels, beta, max_topN, gamma_lo, gamma_hi)
        neighbor_classes[target] = is_neighbor
        labels[:,target] = labels_one_target

    return neighbor_classes, labels


def get_params(dataset_name, input_type):
    return {'alpha' : ALPHA, 'K' : K, 'gamma_lo' : GAMMA_LO, 'gamma_hi' : GAMMA_HI, 'beta' : BETA, 'max_topN' : MAX_TOPN, 'C' : C_DICT[input_type]}


def load_data(dataset_name, input_type):
    assert(input_type in ['cossims', 'logits'])
    clip_scores, gts, _, __, ___, ____, _____ = _load_data(dataset_name, input_type, False)
    return clip_scores, gts


def make_output_filename(dataset_name, input_type, p):
    os.makedirs(OUT_PARENT_DIR, exist_ok=True)
    return os.path.join(OUT_PARENT_DIR, 'graph_strategy_%s_%s_alpha%s_K%d_gammalo%s_gammahi%s_beta%s_maxTopN%d_C%s.pkl'%(dataset_name.split('_')[0], input_type, str(p['alpha']), p['K'], str(p['gamma_lo']), str(p['gamma_hi']), str(p['beta']), p['max_topN'], str(p['C'])))


#gts should be 2D array
#return gt_conditionals, gt_margs
def compute_gt_stats(gts):
    gt_margs = np.mean(gts, axis=0)
    gt_joints = gts.T @ gts / gts.shape[0]
    gt_conditionals = gt_joints / gt_margs[:,np.newaxis]
    assert(not np.any(np.isnan(gt_margs)))
    assert(not np.any(np.isnan(gt_conditionals)))
    return gt_conditionals, gt_margs


def do_logreg_phase(clip_scores, labels, class_mask, C):
    logreg_labels = np.zeros_like(labels)
    for i in tqdm(range(len(class_mask))):
        if not class_mask[i]:
            continue

        my_clf = LogisticRegression(C=C, max_iter=1000)
        my_clf = Pipeline([('scaler', StandardScaler()), ('clf', my_clf)])
        my_clf.fit(clip_scores, labels[:,i])
        logreg_labels[:,i] = np.squeeze(my_clf.predict_proba(clip_scores)[:,1:])

    return logreg_labels


#just for one class
def compute_precision_and_recall(preds, gts):
    preds_num = preds.astype('float32')
    precision = 100.0 * np.sum(preds_num * gts) / np.sum(preds_num)
    recall = 100.0 * np.sum(preds_num * gts) / np.sum(gts)
    return precision, recall


def do_eval(clip_scores, gts, loner_classes, neighbor_classes, labels, logreg_labels):
    initial_APs = np.zeros((clip_scores.shape[1],))
    label_APs = np.zeros((clip_scores.shape[1],))
    logreg_label_APs = np.zeros((clip_scores.shape[1],))
    label_precisions = np.zeros((clip_scores.shape[1],))
    label_recalls = np.zeros((clip_scores.shape[1],))
    for i in range(clip_scores.shape[1]):
        initial_APs[i] = 100.0 * average_precision(clip_scores[:,i], gts[:,i])
        if not (loner_classes[i] or neighbor_classes[i]):
            continue

        label_APs[i] = 100.0 * average_precision(labels[:,i].astype('float32'), gts[:,i])
        logreg_label_APs[i] = 100.0 * average_precision(logreg_labels[:,i], gts[:,i])
        label_precisions[i], label_recalls[i] = compute_precision_and_recall(labels[:,i], gts[:,i])

    initial_overall_mAP = np.mean(initial_APs)
    final_overall_mAP = np.sum((loner_classes | neighbor_classes).astype('float32') * logreg_label_APs + (~(loner_classes | neighbor_classes)).astype('float32') * initial_APs) / gts.shape[1]
    initial_loner_mAP = np.sum(loner_classes.astype('float32') * initial_APs) / np.sum(loner_classes)
    initial_neighbor_mAP = np.sum(neighbor_classes.astype('float32') * initial_APs) / np.sum(neighbor_classes)
    initial_other_mAP = np.sum((~(loner_classes | neighbor_classes)).astype('float32') * initial_APs) / (gts.shape[1] - np.sum(loner_classes) - np.sum(neighbor_classes))
    label_loner_mAP = np.sum(loner_classes.astype('float32') * label_APs) / np.sum(loner_classes)
    label_neighbor_mAP = np.sum(neighbor_classes.astype('float32') * label_APs) / np.sum(neighbor_classes)
    logreg_label_loner_mAP = np.sum(loner_classes.astype('float32') * logreg_label_APs) / np.sum(loner_classes)
    logreg_label_neighbor_mAP = np.sum(neighbor_classes.astype('float32') * logreg_label_APs) / np.sum(neighbor_classes)
    label_loner_mean_precision = np.sum(loner_classes.astype('float32') * label_precisions) / np.sum(loner_classes)
    label_neighbor_mean_precision = np.sum(neighbor_classes.astype('float32') * label_precisions) / np.sum(neighbor_classes)
    label_loner_mean_recall = np.sum(loner_classes.astype('float32') * label_recalls) / np.sum(loner_classes)
    label_neighbor_mean_recall = np.sum(neighbor_classes.astype('float32') * label_recalls) / np.sum(neighbor_classes)
    eval_dict = {'initial_overall_mAP' : initial_overall_mAP,
                    'final_overall_mAP' : final_overall_mAP,
                    'initial_loner_mAP' : initial_loner_mAP,
                    'initial_neighbor_mAP' : initial_neighbor_mAP,
                    'initial_other_mAP' : initial_other_mAP,
                    'label_loner_mAP' : label_loner_mAP,
                    'label_neighbor_mAP' : label_neighbor_mAP,
                    'label_loner_mean_precision' : label_loner_mean_precision,
                    'label_neighbor_mean_precision' : label_neighbor_mean_precision,
                    'label_loner_mean_recall' : label_loner_mean_recall,
                    'label_neighbor_mean_recall' : label_neighbor_mean_recall,
                    'logreg_label_loner_mAP' : logreg_label_loner_mAP,
                    'logreg_label_neighbor_mAP' : logreg_label_neighbor_mAP}
    eval_dict_brief = copy.deepcopy(eval_dict)
    eval_dict['initial_APs'] = initial_APs
    eval_dict['label_APs'] = label_APs
    eval_dict['label_precisions'] = label_precisions
    eval_dict['label_recalls'] = label_recalls
    eval_dict['logreg_label_APs'] = logreg_label_APs
    return eval_dict, eval_dict_brief


def graph_strategy(dataset_name, input_type):
    p = get_params(dataset_name, input_type)
    output_filename = make_output_filename(dataset_name, input_type, p)
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    print('data...')
    clip_scores, gts = load_data(dataset_name, input_type) #clip_scores and gts will both be 2D arrays
    print('gt_stats...')
    gt_conditionals, gt_margs = compute_gt_stats(gts)
    print('loners...')
    loner_classes, labels = get_loner_classes_and_labels(clip_scores, gt_conditionals, gt_margs, p['alpha'], p['K'], p['gamma_lo'], p['gamma_hi'])
    print('%d loner classes'%(np.sum(loner_classes)))
    print('neighbors...')
    neighbor_classes, labels = get_neighbor_classes_and_labels(clip_scores, gt_conditionals, gt_margs, loner_classes, labels, p['beta'], p['max_topN'], p['gamma_lo'], p['gamma_hi'])
    print('%d neighbor classes'%(np.sum(neighbor_classes)))
    print('logreg...')
    logreg_labels = do_logreg_phase(clip_scores, labels, (loner_classes | neighbor_classes), p['C'])
    print('eval...')
    eval_dict, eval_dict_brief = do_eval(clip_scores, gts, loner_classes, neighbor_classes, labels, logreg_labels)
    results = {'classnames' : classnames, 'dataset_name' : dataset_name, 'input_type' : input_type, 'loner_classes' : loner_classes, 'neighbor_classes' : neighbor_classes, 'labels' : labels, 'logreg_labels' : logreg_labels, 'eval_dict' : eval_dict, 'eval_dict_brief' : eval_dict_brief, 'params' : p, 'clip_scores' : clip_scores, 'gts' : gts}
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    print(eval_dict_brief)


def usage():
    print('Usage: python graph_strategy.py <dataset_name> <input_type>')


if __name__ == '__main__':
    graph_strategy(*(sys.argv[1:]))
