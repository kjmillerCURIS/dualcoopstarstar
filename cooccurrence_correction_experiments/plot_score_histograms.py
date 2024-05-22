import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu as threshold_otsu_fn
from tqdm import tqdm
from compute_initial_cossims import PSEUDOLABEL_COSSIMS_FILENAME_DICT
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, get_data_manager


NUM_KDE_POINTS = 500
XLIM = {'cossim' : (0.2, 0.55), 'prob' : (-0.1, 1.1), 'logit' : (-25, 10)}


def compute_TPs_and_TNs_and_accs(scores, targets):
    FPs, TPs, thresholds = roc_curve(targets, scores)
    TNs = 1 - FPs
    accs = (np.sum(scores) * TPs + np.sum(1 - scores) * TNs) / len(scores)
    return TPs, TNs, accs, thresholds


def compute_GMM_threshold(scores):
    assert(False) #I'm not sure this implementation is correct
    my_gmm = GaussianMixture(n_components=2)
    my_gmm.fit(scores[:,np.newaxis])
    w = my_gmm.weights_
    mu = np.squeeze(my_gmm.means_)
    invsigmasq = 1 / np.squeeze(my_gmm.covariances_)
    A = 0.5 * (invsigmasq[1] - invsigmasq[0])
    B = mu[0] * invsigmasq[0] - mu[1] * invsigmasq[1]
    C = 0.5 * (mu[1] ** 2 * invsigmasq[1] - mu[0] ** 2 * invsigmasq[0]) - np.log(w[1]) + np.log(w[0]) + 0.5 * (np.log(invsigmasq[1]) - np.log(invsigmasq[0]))
    solutions = [(-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A), (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)]
    import pdb
    pdb.set_trace()
    if solutions[0] < np.amax(mu) and solutions[0] > np.amin(mu):
        assert(not (solutions[1] < np.amax(mu) and solutions[1] > np.amin(mu)))
        return solutions[0]
    elif solutions[1] < np.amax(mu) and solutions[1] > np.amin(mu):
        assert(not (solutions[0] < np.amax(mu) and solutions[0] > np.amin(mu)))
        return solutions[1]
    else:
        assert(False)


def compute_Otsu_threshold(scores):
    return threshold_otsu_fn(scores[np.newaxis,:])


#assumes that kde_xs is evenly spaced
def compute_modes(kde_xs, kde_ys):
    neg_mode = kde_xs[np.argmax(kde_ys)]
    inflection_index = np.argmin(kde_ys[1:] - kde_ys[:-1])
    right_xs, right_ys = kde_xs[inflection_index:], kde_ys[inflection_index:]
    pos_mode_index = np.argmin(right_ys[2:] + right_ys[:-2] - right_ys[1:-1]) + 1
    pos_mode = right_xs[pos_mode_index]
    return neg_mode, kde_xs[inflection_index]


def compute_best_thresholds(scores, targets, kde_xs, kde_ys):
    precision, recall, thresholds_PR = precision_recall_curve(targets, scores)
    precision, recall = precision[:-1], recall[:-1]
    threshold_dict = {}
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    PR_corner_sqdists = np.square(1 - precision) + np.square(1 - recall)
    TPs, TNs, accs, thresholds_ROC = compute_TPs_and_TNs_and_accs(scores, targets)
    ROC_corner_sqdists = np.square(1 - TPs) + np.square(1 - TNs)
    neg_mode, pos_mode = compute_modes(kde_xs, kde_ys)
    threshold_dict['maximize F1 score'] = thresholds_PR[np.argmax(f1_scores)]
    threshold_dict['minimize (1-prec)^2 + (1-rec)^2'] = thresholds_PR[np.argmin(PR_corner_sqdists)]
    threshold_dict['minimize (1-TP)^2 + (1-TN)^2'] = thresholds_ROC[np.argmin(ROC_corner_sqdists)]
    threshold_dict['neg_mode MEOW'] = neg_mode
    threshold_dict['pos_mode MEOW'] = pos_mode
    return threshold_dict


def compute_scores(cossims, logits, score_type):
    if score_type == 'cossim':
        return cossims
    elif score_type == 'prob':
        return 1 / (1 + np.exp(-logits))
    elif score_type == 'logit':
        return logits
    else:
        assert(False)


def compute_kde(scores, score_type):
    xs = np.linspace(XLIM[score_type][0], XLIM[score_type][1], num=NUM_KDE_POINTS)
    my_kde = gaussian_kde(scores)
    ys = my_kde(xs)
    return xs, ys


def plot_score_histogram_one_score_type(cossims, logits, score_type, targets, classname, dataset_name, ax):
    scores = compute_scores(cossims, logits, score_type)
    xs, ys = compute_kde(scores, score_type)
    threshold_dict = compute_best_thresholds(scores, targets, xs, ys)
    ax.plot(xs, ys)
    for t, threshold_type in enumerate(sorted(threshold_dict.keys())):
        ax.axvline(x=threshold_dict[threshold_type], color=['r','b','orange','gray', 'gray'][t], label=threshold_type)

    ylim = ax.get_ylim()
    neg_scores = scores[targets == 0]
    pos_scores = scores[targets == 1]
    ax.scatter(neg_scores, np.random.uniform(ylim[0], ylim[1], len(neg_scores)), s=3, color='k', marker='.', label='gt negatives')
    ax.scatter(pos_scores, np.random.uniform(ylim[0], ylim[1], len(pos_scores)), s=3, color='g', marker='.', label='gt positives')
    ax.legend()
    ax.set_xlabel(score_type)
    ax.set_ylabel('density')
    ax.set_xlim(XLIM[score_type])
    ax.set_ylim(ylim)
    ax.set_title('%s: class "%s" %s KDE with thresholds'%(dataset_name.split('_')[0], classname, score_type))


def get_plot_filename(classname, dataset_name):
    dataset_nickname = dataset_name.split('_')[0]
    classnickname = classname.replace(' ', '')
    return os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/basic_plots/score_histograms/%s/%s_%s_score_histograms.png'%(dataset_nickname, dataset_nickname, classnickname))


def plot_score_histograms_one_class(cossims, logits, targets, classname, dataset_name):
    plt.clf()
    _, axs = plt.subplots(3, 1, figsize=(6.4, 3 * 4.8))
    for score_type, ax in zip(['cossim', 'prob', 'logit'], axs):
        plot_score_histogram_one_score_type(cossims, logits, score_type, targets, classname, dataset_name, ax)

    plt.tight_layout()
    plot_filename = get_plot_filename(classname, dataset_name)
    plt.savefig(plot_filename, dpi=300)
    plt.clf()
    plt.close()


#return cossims, logits, gts as 2D npy arrays, as well as classnames as list
def load_stuff(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(PSEUDOLABEL_COSSIMS_FILENAME_DICT[dataset_name], 'rb') as f:
        cossims = pickle.load(f)

    cossims = np.array([cossims[impath] for impath in sorted(cossims.keys())])
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        logits = pickle.load(f)

    logits = np.array([logits[impath] for impath in sorted(logits.keys())])
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    return cossims, logits, gts, classnames


def plot_score_histograms(dataset_name):
    cossims, logits, gts, classnames = load_stuff(dataset_name)
    for i, classname in tqdm(enumerate(classnames)):
        plot_score_histograms_one_class(cossims[:,i], logits[:,i], gts[:,i], classname, dataset_name)


def usage():
    print('Usage: python plot_score_histograms.py <dataset_name>')


if __name__ == '__main__':
    plot_score_histograms(*(sys.argv[1:]))
