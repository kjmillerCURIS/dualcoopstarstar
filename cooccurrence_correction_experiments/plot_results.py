import os
import sys
import glob
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from compute_mAP import INITIAL_PSEUDOLABELS_MAP_FILENAME


NEG_RESULTS_PREFIX = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training-oracle-negpenalty-results/mscoco_training-oracle-negpenalty-results')
NEGPOS_RESULTS_PREFIX = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training-oracle-negpospenalty-results/mscoco_training-oracle-negpospenalty-results')
PLOT_PREFIX = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/plot')


COLOR_DICT = {'prob_prob' : 'r', 'prob_logit' : 'g', 'prob_stopgrad_logit' : 'b'}
OTHER_COLOR_DICT = {0.3 : 'r', 0.4 : 'g', 0.6 : 'b'}
LINESTYLE_DICT = {0.25 : 'solid', 0.5 : 'dashed', 0.75 : 'dotted'}
OTHER_LINESTYLE_DICT = {-0.25 : 'solid', -0.5 : 'dashed', -0.75 : 'dotted'}


def plot_neg():
    plt.clf()
    with open(INITIAL_PSEUDOLABELS_MAP_FILENAME, 'rb') as f:
        baseline_mAP = pickle.load(f)

    for neg_cost_type in ['prob_prob', 'prob_logit', 'prob_stopgrad_logit']:
        for epsilon in [0.25, 0.5, 0.75]:
            xs = []
            ys = []
            for alpha in [0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
                xs.append(alpha)
                results_filename = NEG_RESULTS_PREFIX + '-alpha%f-epsilon%f-neg_cost_type%s.pkl'%(alpha, epsilon, neg_cost_type)
                with open(results_filename, 'rb') as f:
                    results = pickle.load(f)

                mAP = results['mAP']
                ys.append(mAP)

            plt.plot(xs, ys, color=COLOR_DICT[neg_cost_type], linestyle=LINESTYLE_DICT[epsilon], marker='o', label=neg_cost_type + ', epsilon=%.2f'%(epsilon))

    plt.plot(plt.xlim(), [baseline_mAP, baseline_mAP], color='k', linestyle='dashed', label='baseline')
    plt.legend()
    plt.title('penalize bad cooccurrences')
    plt.ylabel('mAP')
    plt.xlabel('alpha')
    if not os.path.exists(os.path.dirname(PLOT_PREFIX)):
        os.makedirs(os.path.dirname(PLOT_PREFIX))

    plt.savefig(PLOT_PREFIX + '-neg.png')


def plot_negpos():
    plt.clf()
    with open(INITIAL_PSEUDOLABELS_MAP_FILENAME, 'rb') as f:
        baseline_mAP = pickle.load(f)
    
    negpos_cost_type = 'prob_stopgrad_logit'
    epsilon = 0.25
    for alpha in [0.3, 0.4, 0.6]:
        already_flag = False
        for zeta in [-0.25]:#[-0.75, -0.5, -0.25]:
            xs = []
            ys = []
            for beta in [0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
                xs.append(beta)
                results_filename = NEGPOS_RESULTS_PREFIX + '-alpha%f-epsilon%f-beta%f-zeta%f-negpos_cost_type%s.pkl'%(alpha, epsilon, beta, zeta, negpos_cost_type)
                with open(results_filename, 'rb') as f:
                    results = pickle.load(f)

                mAP = results['mAP']
                ys.append(mAP)

            plt.plot(xs, ys, color=OTHER_COLOR_DICT[alpha], linestyle=OTHER_LINESTYLE_DICT[zeta], marker='o', label='alpha=%.4f, zeta=%.2f'%(alpha, zeta))

    plt.plot(plt.xlim(), [baseline_mAP, baseline_mAP], color='k', linestyle='dashed', label='baseline')
    plt.legend()
    plt.title('penalize bad cooccurrences, reward good ones')
    plt.ylabel('mAP')
    plt.xlabel('beta')
    if not os.path.exists(os.path.dirname(PLOT_PREFIX)):
        os.makedirs(os.path.dirname(PLOT_PREFIX))

    plt.savefig(PLOT_PREFIX + '-negpos.png')


if __name__ == '__main__':
    plot_neg()
    plot_negpos()
