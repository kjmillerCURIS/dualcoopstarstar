import os
import sys
import copy
from collections import Counter
import math
import numpy as np
import pickle
import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.decomposition import NMF
from scipy.optimize import minimize, Bounds
from tqdm import tqdm
from harvest_training_gts import get_data_manager
from compute_joint_marg_disagreement import plot_heatmap, CLUSTER_SORT_FILENAME_DICT
from openai import OpenAI
sys.path.append('.')
from openai_utils import OPENAI_API_KEY
sys.path.pop()


DEBUG = False
LLM_MODEL = 'gpt-3.5-turbo-instruct'
#LLM_MODEL = 'gpt-4'
LLM_LIMIT = 20
DEFAULT_MAX_TOKENS = 300
BIGGER_MAX_TOKENS = 600
YESNO_UNSURE_LENGTH = 10
LOCATION_MIN_COUNT_DICT = {'COCO2014_partial' : 2, 'nuswide_partial' : 2, 'VOC2007_partial' : 2}
LOCATION_NUM_CLUSTERS_DICT = {'COCO2014_partial' : 20, 'nuswide_partial' : 20, 'VOC2007_partial' : 20} #intentionally overclustering, since we allow each class to pick mulitple locations
NUM_LOCATIONS_PER_CLASS_DICT = {'COCO2014_partial' : 3, 'nuswide_partial' : 3, 'VOC2007_partial' : 3}
NUM_LOCATIONS_PER_CLASS_HIGH_DICT = {'COCO2014_partial' : 3, 'nuswide_partial' : 3, 'VOC2007_partial' : 3}
NUM_LOCATIONS_PER_CLASS_LOW_DICT = {'COCO2014_partial' : 1, 'nuswide_partial' : 1, 'VOC2007_partial' : 1}
NUM_LOCATIONS_PER_CLASS_REMOVEPOSITIVES_DICT = {'COCO2014_partial' : 1, 'nuswide_partial' : 1, 'VOC2007_partial' : 1}


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments')
LLM_CACHE_FILENAME = os.path.join(BASE_DIR, 'llm_cache_gpt-3.5-turbo-instruct.pkl')
PLOT_FILENAME_EXPECTED_DICT = {'COCO2014_partial' : os.path.join(BASE_DIR, 'llm_plots', 'mscoco_llm_plot_expected.png'), 'nuswide_partial' : os.path.join(BASE_DIR, 'llm_plots', 'nuswide_llm_plot_expected.png'), 'VOC2007_partial' : os.path.join(BASE_DIR, 'llm_plots', 'voc2007_llm_plot_expected.png')}
PLOT_FILENAME_SURPRISE_DICT = {'COCO2014_partial' : os.path.join(BASE_DIR, 'llm_plots', 'mscoco_llm_plot_surprise.png'), 'nuswide_partial' : os.path.join(BASE_DIR, 'llm_plots', 'nuswide_llm_plot_surprise.png'), 'VOC2007_partial' : os.path.join(BASE_DIR, 'llm_plots', 'voc2007_llm_plot_surprise.png')}
PLOT_FILENAME_EXPECTED_VS_SURPRISE_DICT = {'COCO2014_partial' : os.path.join(BASE_DIR, 'llm_plots', 'mscoco_llm_plot_expected_vs_surprise.png'), 'nuswide_partial' : os.path.join(BASE_DIR, 'llm_plots', 'nuswide_llm_plot_expected_vs_surprise.png'), 'VOC2007_partial' : os.path.join(BASE_DIR, 'llm_plots', 'voc2007_llm_plot_expected_vs_surprise.png')}
OUTCOME2COLOR_EXPECTED = {('no', 'no') : 'red', ('no', 'other') : 'orange', ('no', 'yes') : 'yellow', ('other', 'other') : 'green', ('other', 'yes') : 'blue', ('yes', 'yes') : 'black'}
OUTCOME2COLOR_SURPRISE = OUTCOME2COLOR_EXPECTED
OUTCOME2COLOR_EXPECTED_VS_SURPRISE = {('no_expected', 'yes_surprise') : 'yellow', ('yes_expected', 'no_surprise') : 'blue', ('no_expected', 'no_surprise') : 'green', ('yes_expected', 'yes_surprise') : 'red'}
OUTCOME2COLOR_LOC_COOC = {('no_expected', 'yes_surprise') : 'yellow', ('yes_expected', 'no_surprise') : 'blue', ('unconfident ==> positive',) : 'cornflowerblue', ('unconfident ==> negative',) : 'gold'}
OUTCOME2COLOR_LOC_COOC_MISSINGMIDDLE = {('no_expected', 'yes_surprise') : 'yellow', ('yes_expected', 'no_surprise') : 'blue', ('unconfident ==> positive',) : 'cornflowerblue', ('unconfident ==> negative',) : 'gold', ('unconf ==> unconf',) : 'green'}
OUTCOME2COLOR_LOC_COOC_REMOVEPOSITIVES = {('no_expected', 'yes_surprise') : 'yellow', ('yes_expected', 'no_surprise') : 'blue', ('unconf ==> unconf',) : 'green', ('positive ==> unconfident',) : 'teal'}
NA_COLOR = 'grey'
TERNARY_COOCCURRENCE_MAT_FILENAME_CONFONLY_DICT = {k : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/ternary_cooccurrence_mats/%s/expectedsurprise_confonly.pkl'%(k)) for k in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
TERNARY_COOCCURRENCE_MAT_FILENAME_FILLIN_DICT = {k : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/ternary_cooccurrence_mats/%s/expectedsurprise_fillin_%dloc_%dlpc.pkl'%(k, LOCATION_NUM_CLUSTERS_DICT[k], NUM_LOCATIONS_PER_CLASS_DICT[k])) for k in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
TERNARY_COOCCURRENCE_MAT_FILENAME_FILLIN_MISSINGMIDDLE_DICT = {k : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/ternary_cooccurrence_mats/%s/expectedsurprise_fillin_missingmiddle_%dloc_%dlpchigh_%dlpclow.pkl'%(k, LOCATION_NUM_CLUSTERS_DICT[k], NUM_LOCATIONS_PER_CLASS_HIGH_DICT[k], NUM_LOCATIONS_PER_CLASS_LOW_DICT[k])) for k in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
TERNARY_COOCCURRENCE_MAT_FILENAME_REMOVEPOSITIVES_DICT = {k : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/ternary_cooccurrence_mats/%s/expectedsurprise_removepositives_%dloc_%dlpc.pkl'%(k, LOCATION_NUM_CLUSTERS_DICT[k], NUM_LOCATIONS_PER_CLASS_REMOVEPOSITIVES_DICT[k])) for k in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}


#def plot_heatmap(my_heatmap, classnames, plot_suffix, cluster_sort=None, vis_min=None, vis_max=None, is_mask=False, my_title=None, plot_filename):


def query_llm_helper(client, prompts, max_tokens=DEFAULT_MAX_TOKENS, temperature=0., llm_cache=None):
    if llm_cache is not None and all([(prompt, max_tokens, temperature) in llm_cache for prompt in prompts]):
        return [llm_cache[(prompt, max_tokens, temperature)] for prompt in prompts]

    #print('ACTUAL LLM QUERY (not cache)')
    completion = client.completions.create(model=LLM_MODEL, prompt=prompts, temperature=temperature, max_tokens=max_tokens)
    outputs = [c.text for c in completion.choices]

    if llm_cache is not None:
        for prompt, output in zip(prompts, outputs):
            llm_cache[(prompt, max_tokens, temperature)] = output

    return outputs

def query_llm(prompts, max_tokens=DEFAULT_MAX_TOKENS, temperature=0., llm_cache=None):
    client = OpenAI(api_key=OPENAI_API_KEY)
    if DEBUG:
        print('prompts:')
        print(prompts)

    print('querying LLM...')
    print('(%d prompts)'%(len(prompts)))
    if len(prompts) <= LLM_LIMIT:
        return query_llm_helper(client, prompts, max_tokens=max_tokens, temperature=temperature, llm_cache=llm_cache)
    else:
        cur_t = 0
        outputs = []
        for _ in range(int(round(len(prompts) / LLM_LIMIT)) + 10):
            if cur_t >= len(prompts):
                break

            cur_prompts = prompts[cur_t:min(cur_t+LLM_LIMIT,len(prompts))]
            cur_outputs = query_llm_helper(client, cur_prompts, max_tokens=max_tokens, temperature=temperature, llm_cache=llm_cache)
            outputs.extend(cur_outputs)
            cur_t += LLM_LIMIT

    print('done querying LLM!')
    assert(len(outputs) == len(prompts))
    return outputs


def initialize_NMF(X, supervision_mask, n_components, alpha):
    #initialize unsupervised bits to 0.5 and diagonal to 1.0
    #ask sklearn to do NMF on it
    #average together W and H
    X_sup = copy.deepcopy(X)
    X_sup[~supervision_mask] = 0.5
    np.fill_diagonal(X_sup, 1)
    my_NMF = NMF(n_components=n_components, init='nndsvda', alpha_W=alpha, alpha_H='same', l1_ratio=0, beta_loss=2)
    W = my_NMF.fit_transform(X_sup)
    H = my_NMF.components_
    return 0.5 * W + 0.5 * H.T


def NMF_loss(W, X, supervision_mask, n_components, alpha):
    W = np.reshape(W, (X.shape[0], n_components))
    data_loss = np.sum(supervision_mask.astype('int') * np.square(X - W @ W.T))
    reg_loss = alpha * X.shape[0] * np.sum(np.square(W))
    return 0.5 * data_loss + 0.5 * reg_loss


#this will return two npy arrays, where 1 means should-cooccur, 0 means shouldn't-cooccur, and diagonal is np.nan
#first one will be filling-in the unconfident values with NMF predictions
#second one will replace everything with NMF predictions
def do_NMF_completion(ij2outcome_expected_vs_surprise, num_classes, n_components, alpha):
    X = np.nan * np.ones((num_classes, num_classes))
    supervision_mask = np.zeros((num_classes, num_classes), dtype='bool')
    for ij in sorted(ij2outcome_expected_vs_surprise.keys()):
        i,j = ij
        outcome = ij2outcome_expected_vs_surprise[ij]
        if outcome == ('yes_expected', 'no_surprise'):
            X[i,j] = 1
            X[j,i] = 1
            supervision_mask[i,j] = True
            supervision_mask[j,i] = True
        elif outcome == ('no_expected', 'yes_surprise'):
            X[i,j] = 0
            X[j,i] = 0
            supervision_mask[i,j] = True
            supervision_mask[j,i] = True

    W_init = initialize_NMF(X, supervision_mask, n_components, alpha)
    res = minimize(NMF_loss, W_init.flatten(), args=(X, supervision_mask, n_components, alpha), bounds=Bounds(lb=np.zeros_like(W_init.flatten()), ub=np.inf * np.ones_like(W_init.flatten())))
    W = res.x
    W = np.reshape(W, (X.shape[0], n_components))
    X_pred = W @ W.T
    np.fill_diagonal(X_pred, np.nan)
    X_unconf_filled = copy.deepcopy(X)
    X_unconf_filled[~supervision_mask] = X_pred[~supervision_mask]
    return X_unconf_filled, X_pred


def plot_NMF_heatmap(my_heatmap, classnames, cluster_sort, my_title, plot_filename):
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    classnames = copy.deepcopy(classnames)
    my_heatmap = my_heatmap[cluster_sort][:, cluster_sort]
    classnames = [classnames[cs] for cs in cluster_sort]
    cmap = plt.cm.viridis_r
    cmap.set_bad(color='black')
    plt.clf()
    plt.figure(figsize=(12, 10))
    hmap = plt.imshow(my_heatmap, vmin=0, vmax=1, aspect='auto', cmap=cmap)
    plt.xticks(ticks=np.arange(len(classnames)), labels=classnames, rotation=90)
    plt.yticks(ticks=np.arange(len(classnames)), labels=classnames)
    cbar = plt.colorbar(hmap)
    cbar.set_ticks([0, 0.5, 1])
    plt.title(my_title)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.clf()


#return 'yes', 'no', or 'other'
def postprocess_yesno_output(output):
    output = output.replace('.', '').strip().lower()
    if output in ['yes', 'no']:
        return output
    else:
        if len(output) <= YESNO_UNSURE_LENGTH:
            print('weird yesno output "%s"'%(output))
            assert(False)

        return 'other'


#return ij2outcome, which maps from (i,j) to one of 9 "outcomes", e.g. ('yes', 'yes'), ('no', 'yes'), ('no', 'other'), etc.
def common_to_outcome(template, classnames, llm_cache):
    prompts = []
    index_pairs = []
    for i in range(len(classnames) - 1):
        for j in range(i + 1, len(classnames)):
            index_pairs.extend([(i, j), (j, i)])
            prompt_ij = template % (classnames[i], classnames[j])
            prompt_ji = template % (classnames[j], classnames[i])
            prompts.extend([prompt_ij, prompt_ji])

    outputs = query_llm(prompts, llm_cache=llm_cache)
    outputs = [postprocess_yesno_output(output) for output in outputs]
    ij2outcome = {}
    for index_pair, output in zip(index_pairs, outputs):
        ij = tuple(sorted(index_pair))
        if ij not in ij2outcome:
            ij2outcome[ij] = []

        ij2outcome[ij].append(output)

    for ij in sorted(ij2outcome.keys()):
        assert(len(ij2outcome[ij]) == 2)
        ij2outcome[ij] = tuple(sorted(ij2outcome[ij]))

    return ij2outcome


def plot_outcome_map(ij2outcome, outcome2color, classnames, cluster_sort, my_title, plot_filename):
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    classnames = copy.deepcopy(classnames)
    outcome2datum = {outcome : k for k, outcome in enumerate(sorted(outcome2color.keys()))}
    na_datum = len(outcome2color)
    data = na_datum * np.ones((len(classnames), len(classnames)))
    for ij in sorted(ij2outcome.keys()):
        i, j = ij
        datum = outcome2datum[ij2outcome[ij]]
        data[i,j] = datum
        data[j,i] = datum

    data = data[cluster_sort][:, cluster_sort]
    classnames = [classnames[cs] for cs in cluster_sort]
    categories = sorted(outcome2color.keys())
    colors = [outcome2color[outcome] for outcome in categories]
    categories.append('N/A')
    colors.append(NA_COLOR)
    cmap_listed = ListedColormap(colors)
    legend_handles = [mpatches.Patch(color=color, label=str(category)) for category, color in zip(categories, colors)]
    plt.clf()
    plt.figure(figsize=(12, 10))
    plt.imshow(data, aspect='auto', cmap=cmap_listed)
    plt.xticks(ticks=np.arange(len(classnames)), labels=classnames, rotation=90)
    plt.yticks(ticks=np.arange(len(classnames)), labels=classnames)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(my_title)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.clf()


def llm_cooccurrence_pairs_expect_and_surprise(dataset_name):
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    llm_cache = {}
    if os.path.exists(LLM_CACHE_FILENAME):
        with open(LLM_CACHE_FILENAME, 'rb') as f:
            llm_cache = pickle.load(f)

    with open(CLUSTER_SORT_FILENAME_DICT[dataset_name], 'rb') as f:
        cluster_sort = pickle.load(f)

    template_expected = 'Would you expect a %s to be seen in a picture that contains a %s, yes or no?'
    template_surprise = 'Would you be surprised to see a %s in a picture that contains a %s, yes or no?'
    ij2outcome_expected = common_to_outcome(template_expected, classnames, llm_cache)
    ij2outcome_surprise = common_to_outcome(template_surprise, classnames, llm_cache)
    with open(LLM_CACHE_FILENAME, 'wb') as f:
        pickle.dump(llm_cache, f)

    plot_outcome_map(ij2outcome_expected,OUTCOME2COLOR_EXPECTED,classnames,cluster_sort,'"Would you expect..."',PLOT_FILENAME_EXPECTED_DICT[dataset_name])
    plot_outcome_map(ij2outcome_surprise,OUTCOME2COLOR_SURPRISE,classnames,cluster_sort,'"Would you be surprised..."',PLOT_FILENAME_SURPRISE_DICT[dataset_name])
    ij2outcome_expected_vs_surprise = {}
    for ij in sorted(ij2outcome_expected.keys()):
        expected_part = ('no_expected' if ij2outcome_expected[ij] == ('no', 'no') else 'yes_expected')
        surprise_part = ('no_surprise' if ij2outcome_surprise[ij] == ('no', 'no') else 'yes_surprise')
        ij2outcome_expected_vs_surprise[ij] = (expected_part, surprise_part)

    plot_outcome_map(ij2outcome_expected_vs_surprise,OUTCOME2COLOR_EXPECTED_VS_SURPRISE,classnames,cluster_sort,'"expected" vs "surprise"',PLOT_FILENAME_EXPECTED_VS_SURPRISE_DICT[dataset_name])
    save_confonly_ternary_cooccurrence_mat(ij2outcome_expected_vs_surprise, classnames, dataset_name)
    #do_NMF_experiments(ij2outcome_expected_vs_surprise, classnames, cluster_sort)
    do_location_experiment(ij2outcome_expected_vs_surprise, classnames, cluster_sort, dataset_name, llm_cache=llm_cache)
    do_location_experiment_missingmiddle(ij2outcome_expected_vs_surprise, classnames, cluster_sort, dataset_name, llm_cache=llm_cache)
    do_location_experiment_removepositives(ij2outcome_expected_vs_surprise, classnames, cluster_sort, dataset_name, llm_cache=llm_cache)


##gives a binary vector
#def postprocess_class_location_output(output, locations):
#    ss = output.split('\n')
#    bin_vec = np.zeros((len(locations),))
#    for sss in ss:
#        pred_loc = sss.replace('.','').replace('-','').strip().lower()
#        if pred_loc == '':
#            continue
#
#        if pred_loc not in locations:
#            print('stray location: %s'%(pred_loc))
#            continue
#
#        bin_vec[locations.index(pred_loc)] = 1
#
#    return bin_vec


#gives a binary vector
def postprocess_class_location_output(output, locations):
    bin_vec = np.zeros((len(locations),))
    output = output.replace('\n', ' ').replace('.',' ').replace('-',' ').strip().lower()
    output = ' ' + output + ' '
    for i, location in enumerate(locations):
        if ' ' + location + ' ' in output:
            bin_vec[i] = 1

    return bin_vec


def plot_class_location_map(class_location_map, classnames, cluster_sort, locations, my_title, plot_filename):
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    class_location_map = copy.deepcopy(class_location_map)
    classnames = copy.deepcopy(classnames)
    class_location_map = class_location_map[cluster_sort,:]
    classnames = [classnames[cs] for cs in cluster_sort]
    plt.clf()
    plt.figure(figsize=(6, 10))
    hmap = plt.imshow(class_location_map, aspect='auto')
    plt.xticks(ticks=np.arange(len(locations)), labels=locations, rotation=90)
    plt.yticks(ticks=np.arange(len(classnames)), labels=classnames)
    plt.title(my_title)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.clf()


def compute_class_location_map(ij2outcome_expected_vs_surprise, location_num_clusters, num_locations_per_class, classnames, cluster_sort, dataset_name, llm_cache=None):
    #take confident positives
    confident_positives = [ij for ij in sorted(ij2outcome_expected_vs_surprise.keys()) if ij2outcome_expected_vs_surprise[ij] == ('yes_expected', 'no_surprise')]
    
    #use confident positives to get common locations
    prompts = []
    location_commonality_template = 'Where would you expect to see a %s and a %s together? Please try to keep the answer to roughly one word.'
    for ij in confident_positives:
        i,j = ij
        prompts.append(location_commonality_template % (classnames[i], classnames[j]))
        prompts.append(location_commonality_template % (classnames[j], classnames[i]))

    outputs = query_llm(prompts, llm_cache=llm_cache)
    with open(LLM_CACHE_FILENAME, 'wb') as f:
        pickle.dump(llm_cache, f)

    outputs = [s.replace('.','').strip().lower() for s in outputs]
    location_counter = Counter(outputs)
    print(location_counter)

    #filter locations
    for k in sorted(location_counter.keys()):
        if location_counter[k] < LOCATION_MIN_COUNT_DICT[dataset_name]:
            del location_counter[k]

    print(location_counter)
    location_place_filter_template = 'Is a %s an object or place? Please answer with one word.'
    prompts = [location_place_filter_template % (k) for k in sorted(location_counter.keys())]
    outputs = query_llm(prompts, llm_cache=llm_cache)
    with open(LLM_CACHE_FILENAME, 'wb') as f:
        pickle.dump(llm_cache, f)

    for k, output in zip(sorted(location_counter.keys()), outputs):
        s = output.replace('.','').strip().lower()
        if s != 'place':
            print('LLM thinks a %s is a %s'%(k, s))
            del location_counter[k]

    print(location_counter)
    #(sort by most to least common just so we have some sort of consistent, non-arbitrary ordering)
    locations = [p[0] for p in location_counter.most_common(location_num_clusters)]
    print(locations)

    #figure out which classes go with which locations
    prompts = []
    if num_locations_per_class > 1:
        class_location_template = 'In which %d'%(num_locations_per_class) + ' places are you most likely to find a %s? Please choose ' + '%d of: %s'%(num_locations_per_class, ', '.join(locations))
    else:
        assert(num_locations_per_class == 1)
        class_location_template = 'In which place are you most likely to find a %s? Please choose one of: ' + ', '.join(locations) + ' and keep your answer to roughly one word.'

    for classname in classnames:
        prompts.append(class_location_template % (classname))

    outputs = query_llm(prompts, llm_cache=llm_cache)
    with open(LLM_CACHE_FILENAME, 'wb') as f:
        pickle.dump(llm_cache, f)

    class_location_map = []
    for output in outputs:
        bin_vec = postprocess_class_location_output(output, locations)
        class_location_map.append(bin_vec)

    class_location_map = np.array(class_location_map)
    plot_class_location_map(class_location_map, classnames, cluster_sort, locations, 'class-location map', os.path.join(BASE_DIR, 'llm_plots', '%s_class_location_map_%d_%d.png'%(dataset_name, location_num_clusters, num_locations_per_class)))
    return class_location_map


def do_location_experiment(ij2outcome_expected_vs_surprise, classnames, cluster_sort, dataset_name, llm_cache=None):
    #get common locations and map classes to them
    class_location_map = compute_class_location_map(ij2outcome_expected_vs_surprise, LOCATION_NUM_CLUSTERS_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_DICT[dataset_name], classnames, cluster_sort, dataset_name, llm_cache=llm_cache)

    #and use common locations to decide cooccurrences
    loc_cooc_map = (class_location_map @ class_location_map.T > 0)
    ij2outcome_loc_cooc = {}
    for ij in sorted(ij2outcome_expected_vs_surprise.keys()):
        existing_outcome = ij2outcome_expected_vs_surprise[ij]
        if existing_outcome in [('yes_expected', 'no_surprise'), ('no_expected', 'yes_surprise')]:
            ij2outcome_loc_cooc[ij] = existing_outcome
        else:
            i,j = ij
            ij2outcome_loc_cooc[ij] = ('unconfident ==> positive' if loc_cooc_map[i,j] else 'unconfident ==> negative',)

    plot_outcome_map(ij2outcome_loc_cooc, OUTCOME2COLOR_LOC_COOC, classnames, cluster_sort, 'unconfidents filled in via location', os.path.join(BASE_DIR, 'llm_plots', '%s_llm_plot_loc_cooc_%d_%d.png'%(dataset_name, LOCATION_NUM_CLUSTERS_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_DICT[dataset_name])))
    save_fillin_ternary_cooccurrence_mat(ij2outcome_loc_cooc, classnames, dataset_name)


def do_location_experiment_missingmiddle(ij2outcome_expected_vs_surprise, classnames, cluster_sort, dataset_name, llm_cache=None):
    #get common locations and map classes to them
    class_location_map_high = compute_class_location_map(ij2outcome_expected_vs_surprise, LOCATION_NUM_CLUSTERS_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_HIGH_DICT[dataset_name], classnames, cluster_sort, dataset_name, llm_cache=llm_cache)
    class_location_map_low = compute_class_location_map(ij2outcome_expected_vs_surprise, LOCATION_NUM_CLUSTERS_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_LOW_DICT[dataset_name], classnames, cluster_sort, dataset_name, llm_cache=llm_cache)

    #and use common locations to decide cooccurrences
    loc_cooc_map_high = (class_location_map_high @ class_location_map_high.T > 0)
    loc_cooc_map_low = (class_location_map_low @ class_location_map_low.T > 0)
    ij2outcome_loc_cooc = {}
    for ij in sorted(ij2outcome_expected_vs_surprise.keys()):
        existing_outcome = ij2outcome_expected_vs_surprise[ij]
        if existing_outcome in [('yes_expected', 'no_surprise'), ('no_expected', 'yes_surprise')]:
            ij2outcome_loc_cooc[ij] = existing_outcome
        else:
            i,j = ij
            if loc_cooc_map_low[i,j] and loc_cooc_map_high[i,j]:
                ij2outcome_loc_cooc[ij] = ('unconfident ==> positive',)
            elif (not loc_cooc_map_low[i,j]) and (not loc_cooc_map_high[i,j]):
                ij2outcome_loc_cooc[ij] = ('unconfident ==> negative',)
            else:
                ij2outcome_loc_cooc[ij] = ('unconf ==> unconf',)

    plot_outcome_map(ij2outcome_loc_cooc, OUTCOME2COLOR_LOC_COOC_MISSINGMIDDLE, classnames, cluster_sort, 'unconfidents partially filled in via location', os.path.join(BASE_DIR, 'llm_plots', '%s_llm_plot_loc_cooc_missingmiddle_%d_%d_%d.png'%(dataset_name, LOCATION_NUM_CLUSTERS_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_HIGH_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_LOW_DICT[dataset_name])))
    save_fillin_missingmiddle_ternary_cooccurrence_mat(ij2outcome_loc_cooc, classnames, dataset_name)


def do_location_experiment_removepositives(ij2outcome_expected_vs_surprise, classnames, cluster_sort, dataset_name, llm_cache=None):
    #get common locations and map classes to them
    class_location_map = compute_class_location_map(ij2outcome_expected_vs_surprise, LOCATION_NUM_CLUSTERS_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_REMOVEPOSITIVES_DICT[dataset_name], classnames, cluster_sort, dataset_name, llm_cache=llm_cache)

    #and use common locations to decide cooccurrences
    loc_cooc_map = (class_location_map @ class_location_map.T > 0)
    ij2outcome_loc_cooc = {}
    for ij in sorted(ij2outcome_expected_vs_surprise.keys()):
        existing_outcome = ij2outcome_expected_vs_surprise[ij]
        if existing_outcome == ('no_expected', 'yes_surprise'): #confidently negative
            ij2outcome_loc_cooc[ij] = existing_outcome
        elif existing_outcome != ('yes_expected', 'no_surprise'): #unconfident
            ij2outcome_loc_cooc[ij] = ('unconf ==> unconf',)
        else: #confidently positive
            i,j = ij
            ij2outcome_loc_cooc[ij] = (existing_outcome if loc_cooc_map[i,j] else ('positive ==> unconfident',))

    plot_outcome_map(ij2outcome_loc_cooc, OUTCOME2COLOR_LOC_COOC_REMOVEPOSITIVES, classnames, cluster_sort, 'remove confident positives using location', os.path.join(BASE_DIR, 'llm_plots', '%s_llm_plot_loc_cooc_removepositives_%d_%d.png'%(dataset_name, LOCATION_NUM_CLUSTERS_DICT[dataset_name], NUM_LOCATIONS_PER_CLASS_REMOVEPOSITIVES_DICT[dataset_name])))
    save_removepositives_ternary_cooccurrence_mat(ij2outcome_loc_cooc, classnames, dataset_name)


def save_fillin_ternary_cooccurrence_mat(ij2outcome, classnames, dataset_name):
    mat = np.zeros((len(classnames), len(classnames)))
    for ij in sorted(ij2outcome.keys()):
        i,j = ij
        outcome = ij2outcome[ij]
        if outcome in [('yes_expected', 'no_surprise'), ('unconfident ==> positive',)]:
            mat[i,j] = -1
        elif outcome in [('no_expected', 'yes_surprise'), ('unconfident ==> negative',)]:
            mat[i,j] = 1
        else:
            assert(False)

    ternary_cooccurrence_mat = {'mat' : mat, 'classnames' : classnames}
    with open(TERNARY_COOCCURRENCE_MAT_FILENAME_FILLIN_DICT[dataset_name], 'wb') as f:
        pickle.dump(ternary_cooccurrence_mat, f)


def save_fillin_missingmiddle_ternary_cooccurrence_mat(ij2outcome, classnames, dataset_name):
    mat = np.zeros((len(classnames), len(classnames)))
    for ij in sorted(ij2outcome.keys()):
        i,j = ij
        outcome = ij2outcome[ij]
        if outcome in [('yes_expected', 'no_surprise'), ('unconfident ==> positive',)]:
            mat[i,j] = -1
        elif outcome in [('no_expected', 'yes_surprise'), ('unconfident ==> negative',)]:
            mat[i,j] = 1
        elif outcome == ('unconf ==> unconf',):
            mat[i,j] = 0
        else:
            assert(False)

    ternary_cooccurrence_mat = {'mat' : mat, 'classnames' : classnames}
    with open(TERNARY_COOCCURRENCE_MAT_FILENAME_FILLIN_MISSINGMIDDLE_DICT[dataset_name], 'wb') as f:
        pickle.dump(ternary_cooccurrence_mat, f)


def save_confonly_ternary_cooccurrence_mat(ij2outcome, classnames, dataset_name):
    mat = np.zeros((len(classnames), len(classnames)))
    for ij in sorted(ij2outcome.keys()):
        i,j = ij
        outcome = ij2outcome[ij]
        if outcome == ('yes_expected', 'no_surprise'):
            mat[i,j] = -1
        elif outcome == ('no_expected', 'yes_surprise'):
            mat[i,j] = 1

    ternary_cooccurrence_mat = {'mat' : mat, 'classnames' : classnames}
    with open(TERNARY_COOCCURRENCE_MAT_FILENAME_CONFONLY_DICT[dataset_name], 'wb') as f:
        pickle.dump(ternary_cooccurrence_mat, f)


def save_removepositives_ternary_cooccurrence_mat(ij2outcome, classnames, dataset_name):
    mat = np.zeros((len(classnames), len(classnames)))
    for ij in sorted(ij2outcome.keys()):
        i,j = ij
        outcome = ij2outcome[ij]
        if outcome == ('yes_expected', 'no_surprise'):
            mat[i,j] = -1
        elif outcome == ('no_expected', 'yes_surprise'):
            mat[i,j] = 1

    ternary_cooccurrence_mat = {'mat' : mat, 'classnames' : classnames}
    with open(TERNARY_COOCCURRENCE_MAT_FILENAME_REMOVEPOSITIVES_DICT[dataset_name], 'wb') as f:
        pickle.dump(ternary_cooccurrence_mat, f)


def do_NMF_experiments(ij2outcome_expected_vs_surprise, classnames, cluster_sort):
    for n_components in [5, 10, 20]:
        for alpha in [0.0, 0.001, 0.01, 0.1, 1.0]:
            X_unconf_filled, X_pred = do_NMF_completion(ij2outcome_expected_vs_surprise, len(classnames), n_components, alpha)
            plot_NMF_heatmap(X_unconf_filled, classnames, cluster_sort, 'X_unconf_filled, n_components=%d, alpha=%.5f'%(n_components, alpha), os.path.join(BASE_DIR, 'llm_plots', 'NMF_plot_X_unconf_filled_ncomp%d_alpha%.5f.png'%(n_components, alpha)))
            plot_NMF_heatmap(X_pred, classnames, cluster_sort, 'X_pred, n_components=%d, alpha=%.5f'%(n_components, alpha), os.path.join(BASE_DIR, 'llm_plots', 'NMF_plot_X_pred_ncomp%d_alpha%.5f.png'%(n_components, alpha)))


#def llm_cooccurrence_pairscore(surprise=False):
#    dm = get_data_manager()
#    classnames = dm.dataset.classnames
#    prompts = []
#    index_pairs = []
#    for i in range(len(classnames) - 1):
#        for j in range(i + 1, len(classnames)):
#            index_pairs.extend([(i, j), (j, i)])
#            if surprise:
#                template = 'Would you be surprised to see a %s in a picture that contains a %s, yes or no?'
#            else:
#                template = 'Would you expect a %s to be seen in a picture that contains a %s, yes or no?'
#
#            prompt_ij = template % (classnames[i], classnames[j])
#            prompt_ji = template % (classnames[j], classnames[i])
#            prompts.extend([prompt_ij, prompt_ji])
#
#    outputs = query_llm(prompts)
#    scores = [process_yesno_output(output, surprise=surprise) for output in outputs]
#    my_heatmap = build_pairscore_matrix(index_pairs, scores, len(classnames))
#    with open(CLUSTER_SORT_FILENAME, 'rb') as f:
#        cluster_sort = pickle.load(f)
#
#    if surprise:
#        plot_heatmap(my_heatmap,classnames,None,cluster_sort=cluster_sort,vis_min=-1,vis_max=1,plot_filename='pairscore_temp0_promptSurprise.png')
#    else:
#        plot_heatmap(my_heatmap,classnames,None,cluster_sort=cluster_sort,vis_min=-1,vis_max=1,plot_filename='pairscore_temp0_promptA.png')


def llm_cooccurrence(dataset_name):
#    llm_cooccurrence_pairscore(surprise=True)
    llm_cooccurrence_pairs_expect_and_surprise(dataset_name)


if __name__ == '__main__':
    llm_cooccurrence(*(sys.argv[1:]))
#    random.seed(0)
#    dm = get_data_manager()
#    classnames = dm.dataset.classnames
#    #prompts = ['please partition the following list of object names into groups of objects that would often co-occur together in a visual scene or picture: %s'%(','.join(['"' + classname + '"' for classname in random.sample(classnames, len(classnames))]))]
#    prompts = ['please semantically sort the following object names: %s'%(','.join(['"' + classname + '"' for classname in random.sample(classnames, len(classnames))]))]
#    outputs = query_llm(prompts)
#    #outputs = [postprocess_groupings(output) for output in outputs]
#    import pdb
#    pdb.set_trace()
