import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from noise_model import RESULT_DICT_FILENAME


MODEL_TYPE_LIST = ['constant_f', 'AND_only', 'OR_only', 'additive', 'OR_with_AND_bonus', 'OR_with_AND_bonus_multidelta', 'lookup_table']
PLOT_FILENAME = os.path.splitext(RESULT_DICT_FILENAME)[0] + '.png'


def plot_noise_model_results():
    with open(RESULT_DICT_FILENAME, 'rb') as f:
        d = pickle.load(f)

    plt.clf()
    plt.figure(figsize=(3*6.4,3*4.8))
    plt.title('Mean Fraction of Variance Unexplained (mFVU) for different noise models')
    plt.ylabel('mFVU (lower is better)')
    mFVU_list = [d['experiments'][model_type]['eval_dict']['mFVU'] for model_type in MODEL_TYPE_LIST]
    plt.bar(5*np.arange(len(MODEL_TYPE_LIST)), mFVU_list, width=10*0.5, color='lightblue', edgecolor='black')
    for x, mFVU in zip(5 * np.arange(len(MODEL_TYPE_LIST)), mFVU_list):
        plt.text(x, mFVU + 0.01, '%.3f'%(mFVU), ha='center', va='bottom')

    #plt.boxplot(np.array([d['experiments'][model_type]['eval_dict']['FVUs'] for model_type in MODEL_TYPE_LIST]).T, positions=5*np.arange(len(MODEL_TYPE_LIST)), widths=10*0.3, patch_artist=True, boxprops=dict(facecolor='lightgray', color='black'))
    plt.xticks(ticks=5*np.arange(len(MODEL_TYPE_LIST)), labels=MODEL_TYPE_LIST)
    plt.ylim((0,1))
    plt.savefig(PLOT_FILENAME)
    plt.clf()


def usage():
    print('Usage: python plot_noise_model_results.py')


if __name__ == '__main__':
    plot_noise_model_results()
