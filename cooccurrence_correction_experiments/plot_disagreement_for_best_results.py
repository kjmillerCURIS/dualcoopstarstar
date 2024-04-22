import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from compute_joint_marg_disagreement import compute_joint_marg_disagreement, plot_heatmap, CLUSTER_SORT_FILENAME, VIS_MIN, VIS_MAX
from tune_pseudolabels_negpenalty import RESULTS_PREFIX as RESULTS_PREFIX_NEG
from tune_pseudolabels_negpospenalty import RESULTS_PREFIX as RESULTS_PREFIX_NEGPOS
from harvest_mscoco_training_gts import get_data_manager


NEG_SEARCH = {'alpha' : [0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8], 'epsilon' : [0.25, 0.5, 0.75], 'neg_cost_type' : ['prob_prob', 'prob_logit', 'prob_stopgrad_logit']}
NEGPOS_SEARCH = {'alpha' : [0.3, 0.4, 0.6], 'epsilon' : [0.25], 'beta' : [0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8], 'zeta' : [-0.75, -0.5, -0.25], 'negpos_cost_type' : ['prob_stopgrad_logit']}


def search_generator(experiment_type):
    if experiment_type == 'neg':
        for alpha in NEG_SEARCH['alpha']:
            for epsilon in NEG_SEARCH['epsilon']:
                for neg_cost_type in NEG_SEARCH['neg_cost_type']:
                    d = {'alpha' : alpha, 'epsilon' : epsilon, 'neg_cost_type' : neg_cost_type}
                    results_filename = RESULTS_PREFIX_NEG + '-alpha%f-epsilon%f-neg_cost_type%s.pkl'%(alpha, epsilon, neg_cost_type)
                    yield d, results_filename

    elif experiment_type == 'negpos':
        for alpha in NEGPOS_SEARCH['alpha']:
            for epsilon in NEGPOS_SEARCH['epsilon']:
                for beta in NEGPOS_SEARCH['beta']:
                    for zeta in NEGPOS_SEARCH['zeta']:
                        for negpos_cost_type in NEGPOS_SEARCH['negpos_cost_type']:
                            d = {'alpha':alpha, 'epsilon':epsilon, 'beta':beta, 'zeta':zeta, 'negpos_cost_type':negpos_cost_type}
                            results_filename = RESULTS_PREFIX_NEGPOS + '-alpha%f-epsilon%f-beta%f-zeta%f-negpos_cost_type%s.pkl'%(alpha, epsilon, beta, zeta, negpos_cost_type)
                            yield d, results_filename

    else:
        assert(False)


#experiment_type should be "neg" or "negpos"
def plot_disagreement_common(experiment_type):
    my_search = search_generator(experiment_type)
    best_mAP = float('-inf')
    best_d = None
    best_tuned_logits = None
    for d, results_filename in my_search:
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)

        mAP = results['mAP']
        if mAP > best_mAP:
            best_mAP = mAP
            best_d = d
            best_tuned_logits = results['tuned_logits']

    disagreement = compute_joint_marg_disagreement({impath : 1 / (1 + np.exp(-best_tuned_logits[impath])) for impath in sorted(best_tuned_logits.keys())})
    with open(CLUSTER_SORT_FILENAME, 'rb') as f:
        cluster_sort = pickle.load(f)

    dm = get_data_manager()
    classnames = dm.dataset.classnames
    plot_heatmap(disagreement,classnames,'best_mAP-%s.png'%(experiment_type),cluster_sort=cluster_sort,vis_min=VIS_MIN,vis_max=VIS_MAX,my_title=str(d))


def plot_disagreement_for_best_results():
    plot_disagreement_common('neg')
    plot_disagreement_common('negpos')


if __name__ == '__main__':
    plot_disagreement_for_best_results()
