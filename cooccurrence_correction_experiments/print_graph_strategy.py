import os
import sys
import numpy as np
import pickle
from tqdm import tqdm


def print_graph_strategy_one_class_mask(results, class_mask, classnames):
    for i in range(len(class_mask)):
        if not class_mask[i]:
            continue

        print('')
        print('"%s": intial_AP=%f, label_precision=%f, label_recall=%f, label_AP=%f, logreg_label_AP=%f'%(classnames[i], results['eval_dict']['initial_APs'][i], results['eval_dict']['label_precisions'][i], results['eval_dict']['label_recalls'][i], results['eval_dict']['label_APs'][i], results['eval_dict']['logreg_label_APs'][i]))


def print_graph_strategy(results_filename):
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    classnames = results['classnames']
    print('')
    print('%d loner classes'%(np.sum(results['loner_classes'])))
    print_graph_strategy_one_class_mask(results, results['loner_classes'], classnames)
    print('')
    print('%d neighbor classes'%(np.sum(results['neighbor_classes'])))
    print_graph_strategy_one_class_mask(results, results['neighbor_classes'], classnames)


def usage():
    print('Usage: python print_graph_strategy.py <results_filename>')


if __name__ == '__main__':
    print_graph_strategy(*(sys.argv[1:]))
