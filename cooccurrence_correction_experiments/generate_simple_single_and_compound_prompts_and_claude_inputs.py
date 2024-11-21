import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
from harvest_training_gts import get_data_manager, TRAINING_GTS_FILENAME_DICT


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/second_max_experiments')
PAIR_THRESHOLD = 0.05
TRIPLET_THRESHOLD = 0.025


def get_pairs_and_triplets(gts_arr, num_classes, pair_threshold=PAIR_THRESHOLD, triplet_threshold=TRIPLET_THRESHOLD):
    assert(len(gts_arr.shape) == 2)
    assert(gts_arr.shape[1] == num_classes)
    marginals = np.mean(gts_arr, axis=0)
    assert(marginals.shape == (num_classes,))
    joint_ij = gts_arr.T @ gts_arr / gts_arr.shape[0] #Pr(i,j)
    assert(joint_ij.shape == (num_classes, num_classes))
    j_cond_i = joint_ij / marginals[:,np.newaxis] #Pr(j|i)
    pairs = []
    for i in range(num_classes - 1):
        for j in range(i + 1, num_classes):
            if j_cond_i[i,j] > pair_threshold:
                pairs.append((i,j))

    triplets = []
    triplet2prob = {}
    for i,j in pairs:
        #masked_arr[t,k] == 1 iff (gts_arr[t,i] == 1 and gts_arr[t,j] == 1 and gts_arr[t,k] == 1)
        masked_arr = (gts_arr[:,i] * gts_arr[:,j])[:,np.newaxis] * gts_arr
        assert(masked_arr.shape == gts_arr.shape)
        joint_ijk = np.mean(masked_arr, axis=0) #joint_ijk[k] = Pr(i,j,k)
        best_k = None
        best_prob = float('-inf') #Pr(k|i,j)
        for k in range(num_classes):
            if k in [i,j]:
                continue

            prob = joint_ijk[k] / joint_ij[i,j]
            if prob > triplet_threshold and prob > best_prob:
                best_prob = prob
                best_k = k

        if best_k is not None:
            triplet2prob[(i,j,best_k)] = prob
            if (i,best_k,j) in triplet2prob:
                print('!')
                if prob > triplet2prob[(i,best_k,j)]:
                    triplets = [t for t in triplets if t != (i,best_k,j)]
                    triplets.append((i,j,best_k))
            else:
                triplets.append((i,j,best_k))

    return pairs, triplets


#NOTE: we use the *train* gt distribution to figure out cooccurrence probs, even if we're generating prompts to be used on the test set
def generate_simple_single_and_compound_prompts_and_claude_inputs(dataset_name):
    simple_single_and_compound_prompts_without_claude_filename = os.path.join(BASE_DIR, dataset_name.split('_')[0] + '_test', '%s_simple_single_and_compound_prompts_without_claude.txt'%(dataset_name.split('_')[0]))
    claude_inputs_filename = os.path.join(BASE_DIR, dataset_name.split('_')[0] + '_test', '%s_claude_inputs.txt'%(dataset_name.split('_')[0]))
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    gts_arr = np.array([gts[impath] for impath in sorted(gts.keys())])
    pairs, triplets = get_pairs_and_triplets(gts_arr, len(classnames))
    uncovered_singles = [i for i in range(len(classnames)) if not (any([i in p for p in pairs]) or any([i in t for t in triplets]))]
    print('%d uncovered singles: %s'%(len(uncovered_singles), ', '.join([classnames[i] for i in uncovered_singles])))
    f.close()
    f = open(simple_single_and_compound_prompts_without_claude_filename, 'w')
    for classname in classnames:
        f.write(classname + '\n')

    for i,j in pairs:
        f.write(classnames[i] + ' and ' + classnames[j] + '\n')

    for i,j,k in triplets:
        f.write(classnames[i] + ', ' + classnames[j] + ', and ' + classnames[k] + '\n')

    f.close()
    f = open(claude_inputs_filename, 'w')
    for i,j in pairs:
        f.write(classnames[i] + ' and ' + classnames[j] + '\n')

    for i,j,k in triplets:
        f.write(classnames[i] + ', ' + classnames[j] + ', and ' + classnames[k] + '\n')

    for i in uncovered_singles:
        f.write(classnames[i] + '\n')

    f.close()


def usage():
    print('Usage: python generate_simple_single_and_compound_prompts_and_claude_inputs.py <dataset_name>')


if __name__ == '__main__':
    generate_simple_single_and_compound_prompts_and_claude_inputs(*(sys.argv[1:]))
