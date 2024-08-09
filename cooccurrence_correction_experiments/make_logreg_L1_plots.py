import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from logistic_regression_experiment import make_output_filename, OUT_PARENT_DIR


#def make_output_filename(dataset_name, input_type, standardize, balance, L1, C, miniclass):


C_LIST = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]
SPARSITY_HI_THRESHOLD = 1e-1
SPARSITY_LO_THRESHOLD = 1e-5


#return a list of numbers between 0 and 100, more higher means less sparse
def get_sparsities(logreg_dict, sparsity_threshold):
    sparsities = []
    for my_clf in logreg_dict['model']:
        weights = my_clf.named_steps['clf'].coef_
        sparsities.append(100.0 * np.mean(np.fabs(weights) > sparsity_threshold))

    return sparsities


def get_plot_filename(dataset_name, input_type, standardize, balance):
    os.makedirs(os.path.join(OUT_PARENT_DIR, 'logreg_L1_plots'), exist_ok=True)
    return os.path.join(OUT_PARENT_DIR, 'logreg_L1_plots/logreg_L1_plot_%s_%s_standardize%d_balance%d.png'%(dataset_name.split('_')[0], input_type, standardize, balance))


def make_logreg_L1_oneplot(dataset_name, input_type, standardize, balance):
    Cs = []
    train_mAPs = []
    test_mAPs = []
    min_sparsities_hi = []
    mean_sparsities_hi = []
    max_sparsities_hi = []
    min_sparsities_lo = []
    mean_sparsities_lo = []
    max_sparsities_lo = []
    at_least_one = False
    for C in C_LIST:
        logreg_dict_filename = make_output_filename(dataset_name, input_type, standardize, balance, 1, C, 0)
        if not os.path.exists(logreg_dict_filename):
            print('SKIP: "%s" not exist'%(logreg_dict_filename))
            continue

        at_least_one = True
        Cs.append(C)
        with open(logreg_dict_filename, 'rb') as f:
            logreg_dict = pickle.load(f)

        sparsities_hi = get_sparsities(logreg_dict, SPARSITY_HI_THRESHOLD)
        min_sparsities_hi.append(np.amin(sparsities_hi))
        mean_sparsities_hi.append(np.mean(sparsities_hi))
        max_sparsities_hi.append(np.amax(sparsities_hi))
        sparsities_lo = get_sparsities(logreg_dict, SPARSITY_LO_THRESHOLD)
        min_sparsities_lo.append(np.amin(sparsities_lo))
        mean_sparsities_lo.append(np.mean(sparsities_lo))
        max_sparsities_lo.append(np.amax(sparsities_lo))
        train_mAPs.append(logreg_dict['eval']['train_mAP'])
        test_mAPs.append(logreg_dict['eval']['test_mAP'])
        baseline_train_mAP = logreg_dict['eval']['input_train_mAP']
        baseline_test_mAP = logreg_dict['eval']['input_test_mAP']

    if not at_least_one:
        return

    plt.clf()
    _, axs = plt.subplots(3, figsize=(9, 27))
    axs[0].plot(Cs, train_mAPs, color='r', label='train')
    axs[0].plot(Cs, test_mAPs, color='b', label='test')
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


def make_logreg_L1_plots():
    for dataset_name in ['COCO2014_partial', 'nuswide_partial']:
        for input_type in tqdm(['cossims', 'probs', 'log_probs', 'logits']):
            for standardize in [0, 1]:
                for balance in [0, 1]:
                    make_logreg_L1_oneplot(dataset_name, input_type, standardize, balance)


if __name__ == '__main__':
    make_logreg_L1_plots()
