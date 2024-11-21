import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
DATASET_NAME_LIST = ['COCO2014_partial', 'VOC2007_partial', 'nuswideTheirVersion_partial']
DISP_DICT = {'COCO2014_partial' : 'COCO', 'VOC2007_partial' : 'VOC', 'nuswideTheirVersion_partial' : 'NUSWIDE'}
MODEL_TYPE_LIST = ['ViT-L14336px', 'ViT-L14', 'ViT-B16', 'ViT-B32', 'RN50x64', 'RN50x16', 'RN50x4', 'RN101', 'RN50']


def get_avg_class_APs(dataset_name):
    baseline_APs = {}
    our_APs = {}
    for model_type in MODEL_TYPE_LIST:
        result_filename = os.path.join(BASE_DIR, '%s_test/result_files/%s_test_%s_results.pkl'%(dataset_name.split('_')[0], dataset_name.split('_')[0], model_type))
        with open(result_filename, 'rb') as f:
            d = pickle.load(f)
            classnames = sorted(d['ensemble_single_uncalibrated']['APs'].keys())
            if len(baseline_APs) == 0:
                baseline_APs = {c : [] for c in classnames}
                our_APs = {c : [] for c in classnames}

            for c in classnames:
                baseline_APs[c].append(d['ensemble_single_uncalibrated']['APs'][c])
                our_APs[c].append(d['allpcawsing_avg']['APs'][c])

    baseline_APs = {c : np.mean(baseline_APs[c]) for c in classnames}
    our_APs = {c : np.mean(our_APs[c]) for c in classnames}
    return baseline_APs, our_APs


def make_per_class_plot_one_dataset(dataset_name):
    baseline_APs, our_APs = get_avg_class_APs(dataset_name)
    classnames = sorted(baseline_APs.keys())
    plt.clf()
    plt.figure(figsize=(12, 4.5))
    x = np.arange(len(classnames))
    plt.title('Class APs for our method vs vanilla ZSCLIP - %s'%(DISP_DICT[dataset_name]), fontsize=18)
    plt.xlabel('classname', fontsize=18)
    plt.ylabel('AP (%)', fontsize=18)
    plt.bar(x, -np.array([baseline_APs[c] for c in classnames]), label='ZSCLIP', color='blue')
    plt.bar(x, np.array([our_APs[c] for c in classnames]), label='Ours', color='green')
    plt.scatter(x, [our_APs[c] - baseline_APs[c] for c in classnames], color='k', marker='o')
    plt.xticks(x, classnames, rotation=90, ha='center')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, axis='x', linestyle='--', color='gray', alpha=0.7)  # Vertical gridlines
    plt.grid(True, axis='y', linestyle='--', color='gray', alpha=0.7)  # Horizontal gridlines
    plt.legend(fontsize=18)
    plt.tight_layout()
    plot_filename = os.path.join(BASE_DIR, '%s_test_class_APs.png'%(dataset_name.split('_')[0]))
    plt.savefig(plot_filename)
    plt.clf()


def make_per_class_plots():
    for dataset_name in tqdm(DATASET_NAME_LIST):
        make_per_class_plot_one_dataset(dataset_name)


if __name__ == '__main__':
    make_per_class_plots()
