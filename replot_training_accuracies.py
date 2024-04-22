import os
import sys
import glob
import pickle
from tqdm import tqdm
from compute_training_accuracies import just_plot


#def just_plot(pseudo_mAP, corrected_pseudo_mAP, secondcorrected_pseudo_mAP, secondcorrected_epoch, train_mAPs, train_epochs, title, plot_filename):


def replot_training_accuracies():
    pkl_filenames=sorted(glob.glob('../vislang-domain-exploration-data/dualcoopstarstar-data/plots_and_tables/frozen_pseudolabel_w*.pkl'))
    for pkl_filename in tqdm(pkl_filenames):
        plot_filename = os.path.splitext(pkl_filename)[0] + '.png'
        with open(pkl_filename, 'rb') as f:
            d = pickle.load(f)

        just_plot(d['pseudo_mAP'], d['corrected_pseudo_mAP'], d['secondcorrected_pseudo_mAP'], d['secondcorrected_epoch'], d['train_mAPs'], d['train_epochs'], d['title'], plot_filename)


if __name__ == '__main__':
    replot_training_accuracies()
