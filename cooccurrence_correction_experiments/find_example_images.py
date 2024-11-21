import os
import sys
import numpy as np
import pickle
from tqdm import tqdm


def get_prompt(index, score, d_compound_cossims, impath):
    assert(d_compound_cossims['impaths'][index] == impath)
    cossims = d_compound_cossims['cossims'][index,:]
    match_mask = (np.fabs(cossims - score) < 1e-6)
    assert(np.sum(match_mask) == 1)
    prompt_index = np.nonzero(match_mask)[0][0]
    return d_compound_cossims['simple_single_and_compound_prompts'][prompt_index]


def calibrate_compound_cossims(d_compound_cossims):
    num_classes = d_compound_cossims['gts'].shape[1]
    mean = np.mean(d_compound_cossims['cossims'][:,:num_classes], axis=1, keepdims=True)
    sd = np.std(d_compound_cossims['cossims'][:,:num_classes], axis=1, ddof=1, keepdims=True)
    d_compound_cossims['cossims'] = (d_compound_cossims['cossims'] - mean) / sd
    mean = np.mean(d_compound_cossims['cossims'], axis=0, keepdims=True)
    sd = np.std(d_compound_cossims['cossims'], axis=0, ddof=1, keepdims=True)
    d_compound_cossims['cossims'] = (d_compound_cossims['cossims'] - mean) / sd


def find_example_images(extra_filename, compound_cossims_filename):
    with open(extra_filename, 'rb') as f:
        extra = pickle.load(f)

    with open(compound_cossims_filename, 'rb') as f:
        d_compound_cossims = pickle.load(f)

    calibrate_compound_cossims(d_compound_cossims)

    gts_arr = np.array([row['cat'] for row in extra['gts']])
    single_arr = extra['ensemble_single_scores']['cat']
    first_arr = extra['first_max_scores']['cat']
    second_arr = extra['second_max_scores']['cat']
    impaths = extra['impaths']

    print(np.unique(gts_arr))
    assert(np.all((gts_arr == 0) | (gts_arr == 1)))
    gts_pos_arr = (gts_arr > 0)
    gts_neg_arr = (gts_arr == 0)

    A_pos_B_neg_mask = (gts_pos_arr[:,np.newaxis] & gts_neg_arr[np.newaxis,:])
    single_diffs = single_arr[:,np.newaxis] - single_arr[np.newaxis,:]
    first_diffs = first_arr[:,np.newaxis] - first_arr[np.newaxis,:]
    second_diffs = second_arr[:,np.newaxis] - second_arr[np.newaxis,:]
    quality = -single_diffs - first_diffs + 2.5 * second_diffs + 0.5 * np.random.randn(len(impaths), len(impaths))
    interest_mask = (A_pos_B_neg_mask & (single_diffs < 0) & (first_diffs < 0) & (second_diffs > 0))
    A_indices = np.tile(np.arange(len(impaths))[:,np.newaxis], (1, len(impaths)))
    B_indices = np.tile(np.arange(len(impaths))[np.newaxis,:], (len(impaths), 1))
    print(np.sum(interest_mask))
    quality[~interest_mask] = float('-inf')
    is_best_mask = (quality == np.amax(quality))
    print(A_indices[is_best_mask])
    print(B_indices[is_best_mask])
    print(single_diffs[is_best_mask])
    print(first_diffs[is_best_mask])
    print(second_diffs[is_best_mask])
    A_index = A_indices[is_best_mask][0]
    B_index = B_indices[is_best_mask][0]
    A_impath, B_impath = impaths[A_index], impaths[B_index]
    print('Gt positive:')
    print('impath="%s"'%(A_impath))
    print('single=%f, 1max=%f ("%s"), 2max=%f ("%s")'%(single_arr[A_index], first_arr[A_index], get_prompt(A_index, first_arr[A_index], d_compound_cossims, A_impath), second_arr[A_index], get_prompt(A_index, second_arr[A_index], d_compound_cossims, A_impath)))
    print('')
    print('Gt negative:')
    print('impath="%s"'%(B_impath))
    print('single=%f, 1max=%f ("%s"), 2max=%f ("%s")'%(single_arr[B_index], first_arr[B_index], get_prompt(B_index, first_arr[B_index], d_compound_cossims, B_impath), second_arr[B_index], get_prompt(B_index, second_arr[B_index], d_compound_cossims, B_impath)))


if __name__ == '__main__':
    find_example_images(*(sys.argv[1:]))
