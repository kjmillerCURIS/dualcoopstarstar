import os
import sys
import copy
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from run_compound_calibration_experiments import get_compound_output_filename


#def get_compound_output_filename(dataset_name, model_type, single_probe_type, compound_probe_type, compounding_strategy, calibration_type):


MODEL_TYPE = 'ViT-L14336px'
SINGLE_PROBE_TYPE = 'ensemble_80'
COMPOUND_PROBE_TYPE = 'a_photo_of_a_i_and_a_j'
NUM_EXAMPLE_IMAGES = 300


def make_barplot(sorted_before_APs, sorted_after_APs, sorted_classnames, my_title, plot_filename):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_before_APs))
    ax.bar(x, -np.abs(sorted_before_APs), label='before compounding', color='blue')
    ax.bar(x, np.abs(sorted_after_APs), label='after compounding', color='green')
    ax.scatter(x, sorted_after_APs - sorted_before_APs, color='k', marker='o')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_classnames, rotation=90, ha='center')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.7)  # Vertical gridlines
    ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.7)  # Horizontal gridlines
    ax.set_xlabel('class')
    ax.set_ylabel('AP (%)')
    ax.set_title(my_title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.clf()


#return sorted_before_APs, sorted_after_APs, sorted_classnames
#should put hurt first, help last, in absolute terms
def sort_by_change(before_APs, after_APs, classnames):
    diffs = after_APs - before_APs
    indices = np.argsort(diffs)
    sorted_before_APs = before_APs[indices]
    sorted_after_APs = after_APs[indices]
    sorted_classnames = [classnames[idx] for idx in indices]
    return sorted_before_APs, sorted_after_APs, sorted_classnames, indices


def make_scatterplot(before_scores, after_scores, gts, before_AP, after_AP, my_title, plot_filename):
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    rand_y_pos = np.random.uniform(0, 1, size=np.sum(gts).astype('int64'))
    rand_y_neg = np.random.uniform(0, 1, size=np.sum(1 - gts).astype('int64'))
    before_scores_pos = before_scores[gts == 1]
    before_scores_neg = before_scores[gts == 0]
    after_scores_pos = after_scores[gts == 1]
    after_scores_neg = after_scores[gts == 0]
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 20), sharex=True)
    plt.suptitle(my_title)
    ax1.set_title('before compounding (AP=%f)'%(before_AP))
    ax2.set_title('after compounding (AP=%f)'%(after_AP))
    ax1.set_xlabel('score')
    ax2.set_xlabel('score')
    ax1.set_ylabel('random')
    ax2.set_ylabel('random')
    ax1.scatter(before_scores_neg, rand_y_neg, color='r', marker='.', s=6, label='gt neg')
    ax1.scatter(before_scores_pos, rand_y_pos, color='k', marker='.', s=6, label='gt pos')
    ax1.legend()
    ax2.scatter(after_scores_neg, rand_y_neg, color='r', marker='.', s=6, label='gt neg')
    ax2.scatter(after_scores_pos, rand_y_pos, color='k', marker='.', s=6, label='gt pos')
    ax2.legend()
    plt.savefig(plot_filename)
    plt.clf()


def scatterplot_one_class(before_output_dict, after_output_dict, sorted_before_APs, sorted_after_APs, sorted_classnames, indices, after_output_filename, is_calibrated, best_or_worst):
    index = {'best' : indices[-1], 'worst' : indices[0]}[best_or_worst]
    before_AP = {'best' : sorted_before_APs[-1], 'worst' : sorted_before_APs[0]}[best_or_worst]
    after_AP = {'best' : sorted_after_APs[-1], 'worst' : sorted_after_APs[0]}[best_or_worst]
    before_scores = before_output_dict['scores'][:,index]
    after_scores = after_output_dict['scores'][:,index]
    gts = after_output_dict['input_dict']['gts'][:,index]
    classname = {'best' : sorted_classnames[-1], 'worst' : sorted_classnames[0]}[best_or_worst]
    my_title = '"%s" %s'%(classname, {False : 'without calibration', True : 'with calibration'}[is_calibrated])
    plot_filename = os.path.join(os.path.dirname(after_output_filename), 'analysis', {False : 'without_calibration', True : 'with_calibration'}[is_calibrated], 'scores-%s-'%(classname.replace(' ', '')) + os.path.splitext(os.path.basename(after_output_filename))[0] + '.png')
    make_scatterplot(before_scores, after_scores, gts, before_AP, after_AP, my_title, plot_filename)


def compute_opponent_ranks(scores, gts):
    indices = np.argsort(scores)
    invindices = np.argsort(indices)
    sorted_gts = gts[indices]
    assert(all([sorted_gts[invindices[i]] == gts[i] for i in range(len(gts))]))
    ranks_for_neg = 100.0 * np.cumsum(1.0 * sorted_gts) / np.sum(1.0 * gts)
    ranks_for_pos = 100.0 * np.cumsum(1.0 - sorted_gts) / np.sum(1.0 - gts)
    opponent_ranks = gts * ranks_for_pos[invindices] + (1.0 - gts) * ranks_for_neg[invindices]
    return opponent_ranks


def make_text_bar(s, width):
    numI = np.zeros((100, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (50, 70)  # Bottom-left corner of the text
    font_scale = 1
    color = (0, 255, 0)  # Green color in BGR
    thickness = 1
    cv2.putText(numI, s, position, font, font_scale, color, thickness)
    return numI


def visualize_example_image(image_filename, vis_dir, classname, gt, before_score, after_score, before_opponent_rank, after_opponent_rank, argmax_str):
    os.makedirs(vis_dir, exist_ok=True)
    numI = cv2.imread(image_filename)
    numI = cv2.resize(numI, (3 * numI.shape[1], 3 * numI.shape[0]))
    width = numI.shape[1]
    bar_list = []
    bar_list.append(make_text_bar('"%s" (gt=%d)'%(classname, gt), width))
    bar_list.append(make_text_bar('before compounding: score=%.3f (>%.0f%% of %s)'%(before_score, before_opponent_rank, {0 : 'positives', 1 : 'negatives'}[gt]), width))
    bar_list.append(make_text_bar('after compounding: score=%.3f (>%.0f%% of %s)'%(after_score, after_opponent_rank, {0 : 'positives', 1 : 'negatives'}[gt]), width))
    bar_list.append(make_text_bar('argmax=%s'%(argmax_str), width))
    numIbar = np.vstack(bar_list)
    numIstitch = np.vstack([numIbar, numI])
    cv2.imwrite(os.path.join(vis_dir, os.path.basename(image_filename)), numIstitch)


def get_argmax_str(after_output_dict, image_index, class_index, classnames):
    single_cossims = after_output_dict['input_dict']['single_cossims'][image_index,:]
    compound_cossims = after_output_dict['input_dict']['compound_cossims'][image_index]
    compound_cossims = compound_cossims.toarray() - 1
    single_cossim = single_cossims[class_index]
    compound_cossim_me_and_other = np.amax(compound_cossims[class_index,:])
    compound_cossim_other_and_me = np.amax(compound_cossims[:,class_index])
    if single_cossim >= max(compound_cossim_me_and_other, compound_cossim_other_and_me):
        return 'single("%s")'%(classnames[class_index])
    elif compound_cossim_me_and_other >= compound_cossim_other_and_me:
        other_class_index = np.argmax(compound_cossims[class_index,:])
        return 'compound("%s","%s")'%(classnames[class_index], classnames[other_class_index])
    else:
        other_class_index = np.argmax(compound_cossims[:,class_index])
        return 'compound("%s","%s")'%(classnames[other_class_index], classnames[class_index])


def get_image_filename(after_output_dict, image_index):
    return after_output_dict['input_dict']['impaths'][image_index]


#before_opponent_ranks and after_opponent_ranks should be in normal order
def visualize_image_examples_oneclass_onegt(before_output_dict, after_output_dict, classnames, class_index, gt, image_indices, before_opponent_ranks, after_opponent_ranks, after_output_filename, is_calibrated):
    vis_dir = os.path.join(os.path.dirname(after_output_filename), 'analysis', {False : 'without_calibration', True : 'with_calibration'}[is_calibrated], '%s-most_changed_%s-'%(classnames[class_index].replace(' ', ''), {0 : 'negatives', 1 : 'positives'}[gt]) + os.path.splitext(os.path.basename(after_output_filename))[0])
    for image_index in tqdm(image_indices):
        image_filename = get_image_filename(after_output_dict, image_index)
        classname = classnames[class_index]
        before_score = before_output_dict['scores'][image_index, class_index]
        after_score = after_output_dict['scores'][image_index, class_index]
        before_opponent_rank = before_opponent_ranks[image_index]
        after_opponent_rank = after_opponent_ranks[image_index]
        argmax_str = get_argmax_str(after_output_dict, image_index, class_index, classnames)
        visualize_example_image(image_filename, vis_dir, classname, gt, before_score, after_score, before_opponent_rank, after_opponent_rank, argmax_str)


def visualize_image_examples_oneclass(before_output_dict, after_output_dict, classnames, sorted_class_indices, after_output_filename, is_calibrated, best_or_worst):
    class_index = {'best' : sorted_class_indices[-1], 'worst' : sorted_class_indices[0]}[best_or_worst]
    before_scores = before_output_dict['scores'][:,class_index]
    after_scores = after_output_dict['scores'][:,class_index]
    gts = after_output_dict['input_dict']['gts'][:,class_index]
    before_opponent_ranks = compute_opponent_ranks(before_scores, gts)
    after_opponent_ranks = compute_opponent_ranks(after_scores, gts)
    opponent_rank_diffs = after_opponent_ranks - before_opponent_ranks
    pos_thingies = [(diff, i) for i, diff in enumerate(opponent_rank_diffs) if gts[i] == 1]
    neg_thingies = [(diff, i) for i, diff in enumerate(opponent_rank_diffs) if gts[i] == 0]
    sorted_pos_thingies = sorted(pos_thingies, key=lambda thingy: thingy[0], reverse=True)
    sorted_neg_thingies = sorted(neg_thingies, key=lambda thingy: thingy[0], reverse=True)
    pos_image_indices = [thingy[1] for thingy in sorted_pos_thingies[:min(NUM_EXAMPLE_IMAGES, len(sorted_pos_thingies))]]
    neg_image_indices = [thingy[1] for thingy in sorted_neg_thingies[:min(NUM_EXAMPLE_IMAGES, len(sorted_neg_thingies))]]
    visualize_image_examples_oneclass_onegt(before_output_dict, after_output_dict, classnames, class_index, 1, pos_image_indices, before_opponent_ranks, after_opponent_ranks, after_output_filename, is_calibrated)
    visualize_image_examples_oneclass_onegt(before_output_dict, after_output_dict, classnames, class_index, 0, neg_image_indices, before_opponent_ranks, after_opponent_ranks, after_output_filename, is_calibrated)


def plot_class_performance_change(dataset_name, is_calibrated):
    is_calibrated = int(is_calibrated)

    before_output_filename = get_compound_output_filename(dataset_name, MODEL_TYPE, SINGLE_PROBE_TYPE, COMPOUND_PROBE_TYPE, 'no_compounding', {False : 'no_calibration', True : 'standardize'}[is_calibrated])
    after_output_filename = get_compound_output_filename(dataset_name, MODEL_TYPE, SINGLE_PROBE_TYPE, COMPOUND_PROBE_TYPE, 'max_compounding', {False : 'no_calibration', True : 'standardize_using_single_stats'}[is_calibrated])
    with open(before_output_filename, 'rb') as f:
        before_output_dict = pickle.load(f)

    with open(after_output_filename, 'rb') as f:
        after_output_dict = pickle.load(f)

    classnames = before_output_dict['input_dict']['classnames']
    before_APs = before_output_dict['eval_dict']['class_APs']
    after_APs = after_output_dict['eval_dict']['class_APs']
    sorted_before_APs,sorted_after_APs,sorted_classnames,indices = sort_by_change(before_APs,after_APs,classnames)
    my_barplot_title = {False : 'class AP comparison (without calibration)', True : 'class AP comparison (with calibration)'}[is_calibrated]
    barplot_filename = os.path.join(os.path.dirname(after_output_filename), 'analysis', {False : 'without_calibration', True : 'with_calibration'}[is_calibrated], 'class_breakdown_comparison-' + os.path.splitext(os.path.basename(after_output_filename))[0] + '.png')
    make_barplot(sorted_before_APs, sorted_after_APs, sorted_classnames, my_barplot_title, barplot_filename)
    scatterplot_one_class(before_output_dict, after_output_dict, sorted_before_APs, sorted_after_APs, sorted_classnames, indices, after_output_filename, is_calibrated, 'best')
    scatterplot_one_class(before_output_dict, after_output_dict, sorted_before_APs, sorted_after_APs, sorted_classnames, indices, after_output_filename, is_calibrated, 'worst')
    visualize_image_examples_oneclass(before_output_dict, after_output_dict, classnames, indices, after_output_filename, is_calibrated, 'best')
    visualize_image_examples_oneclass(before_output_dict, after_output_dict, classnames, indices, after_output_filename, is_calibrated, 'worst')


def usage():
    print('Usage: python plot_class_performance_change.py <dataset_name> <is_calibrated>')


if __name__ == '__main__':
    plot_class_performance_change(*(sys.argv[1:]))
