import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT
from compute_initial_cossims import PSEUDOLABEL_COSSIMS_FILENAME_DICT


LAMBDA_LIST = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]
SPARSITY_HI_THRESHOLD = 1e-1
SPARSITY_LO_THRESHOLD = 1e-5


#return scores, gts
def load_data(dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    with open(PSEUDOLABEL_COSSIMS_FILENAME_DICT[dataset_name], 'rb') as f:
        scores = pickle.load(f)

    scores = np.array([scores[impath] for impath in sorted(scores.keys())])
    return scores, gts


def standardize_scores(scores):
    assert(False) #KEVIN


#assume scores are standardized
#return W (which is diagonal)
def fit_baseline(scores, gts):
    assert(False) #KEVIN


#assume scores are standardized
#return W
def fit_lasso(scores, gts):
    assert(False) #KEVIN


def print_sparsity_hist(W):
    assert(False) #KEVIN


#return mean_RMSE, mean_R2, min_sparsity_hi, mean_sparsity_hi, max_sparsity_hi, min_sparsity_lo, mean_sparsity_lo, max_sparsity_lo
def eval_W(scores, gts, W):
    assert(False) #KEVIN


def noising_experiment_one_lambda(dataset_name, my_lambda):
    scores, gts = load_data(dataset_name)
    scores = standardize_scores(scores)
    assert(False) #KEVIN


def get_plot_filename(dataset_name):
    assert(False) #KEVIN


def noising_experiment_one_dataset(dataset_name):
    my_lambdas = []
    mean_RMSEs = []
    min_sparsities_hi = []
    mean_sparsities_hi = []
    max_sparsities_hi = []
    min_sparsities_lo = []
    mean_sparsities_lo = []
    max_sparsities_lo = []
    for my_lambda in LAMBDA_LIST:
        min_sparsity_hi, mean_sparsity_hi, max_sparsity_hi, min_sparsity_lo, mean_sparsity_lo, max_sparsity_lo, mean_RMSE, baseline_RMSE = noising_experiment_one_lambda(dataset_name, my_lambda)
        min_sparsities_hi.append(min_sparsity_hi)
        mean_sparsities_hi.append(mean_sparsity_hi)
        max_sparsities_hi.append(max_sparsity_hi)
        min_sparsities_lo.append(min_sparsity_lo)
        mean_sparsities_lo.append(mean_sparsity_lo)
        max_sparsities_lo.append(max_sparsity_lo)
        mean_RMSEs.append(mean_RMSE)
        my_lambdas.append(my_lambda)

    plt.clf()
    _, axs = plt.subplots(3, figsize=(9, 27))
    axs[0].plot(my_lambdas, mean_RMSEs, color='r', label='lasso')
    axs[0].plot(my_lambdas, mean_RMSEs, color='r', linestyle='dotted', label='baseline (only use y_i for hat{y}_i)')
    assert(False) #KEVIN

    my_xlim = axs[0].get_xlim()
    axs[0].plot(my_xlim, [baseline_train_mAP, baseline_train_mAP], color='r', linestyle='dotted', label='baseline_train')
    axs[0].plot(my_xlim, [baseline_test_mAP, baseline_test_mAP], color='b', linestyle='dotted', label='baseline_test')
    ymin = min([baseline_train_mAP, baseline_test_mAP])
    ymax = max([baseline_train_mAP, baseline_test_mAP, np.amax(train_mAPs), np.amax(test_mAPs)])
    axs[0].set_xscale('log')
    axs[0].set_title('mAP vs C')
    axs[0].set_xlabel('C (higher means less regularization)')
    axs[0].set_ylabel('mAP')
    axs[0].legend()
    axs[0].set_ylim((ymin-2, ymax+2))
    axs[1].plot(Cs, min_sparsities_hi, color='r', label='min')
    axs[1].plot(Cs, mean_sparsities_hi, color='k', label='mean')
    axs[1].plot(Cs, max_sparsities_hi, color='b', label='max')
    axs[1].set_xscale('log')
    axs[1].set_title('sparsity vs C')
    axs[1].set_xlabel('C (higher means less regularization)')
    axs[1].set_ylabel('%' + ' weights > %s'%(str(SPARSITY_LO_THRESHOLD)))
    axs[1].legend()
    axs[1].set_ylim((0,100))
    axs[2].plot(Cs, min_sparsities_lo, color='r', label='min')
    axs[2].plot(Cs, mean_sparsities_lo, color='k', label='mean')
    axs[2].plot(Cs, max_sparsities_lo, color='b', label='max')
    axs[2].set_xscale('log')
    axs[2].set_title('sparsity vs C')
    axs[2].set_xlabel('C (higher means less regularization)')
    axs[2].set_ylabel('%' + ' weights > %s'%(str(SPARSITY_LO_THRESHOLD)))
    axs[2].legend()
    axs[2].set_ylim((0,100))
    plt.tight_layout()
    plot_filename = get_plot_filename(dataset_name, input_type, standardize, balance)
    plt.savefig(plot_filename)
    plt.clf()
