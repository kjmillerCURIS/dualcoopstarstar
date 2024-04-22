import os
import sys
import numpy as np
import pickle
import random
from tqdm import tqdm
from collections import Counter


DESCRIPTOR = {-1 : 'YES', 1 : 'NO', 0 : 'MAYBE'}


def sample_llm_mistakes(gt_ternary_filename, llm_ternary_filename, num_samples, random_seed):
    num_samples = int(num_samples)
    random_seed = int(random_seed)
    random.seed(random_seed)

    with open(gt_ternary_filename, 'rb') as f:
        gt_ternary = pickle.load(f)

    with open(llm_ternary_filename, 'rb') as f:
        llm_ternary = pickle.load(f)

    assert(gt_ternary['classnames'] == llm_ternary['classnames'])
    classnames = gt_ternary['classnames']
    gt_mat = gt_ternary['mat']
    llm_mat = llm_ternary['mat']
    assert(np.allclose(gt_mat, np.triu(gt_mat, k=1)))
    assert(np.allclose(llm_mat, np.triu(llm_mat, k=1)))
    diff_ijs = np.array(np.nonzero(llm_mat != gt_mat)).T
    ks = random.sample(range(diff_ijs.shape[0]), min(num_samples, diff_ijs.shape[0]))
    diff_ijs = [diff_ijs[k,:] for k in ks]
    nomaybe_yes_counter = Counter([])
    for ij in diff_ijs:
        i,j = ij
        print('C_gt("%s", "%s") = %s, but C_llm("%s", "%s") = %s'%(classnames[i], classnames[j], DESCRIPTOR[gt_mat[i,j]], classnames[i], classnames[j], DESCRIPTOR[llm_mat[i,j]]))
        if DESCRIPTOR[gt_mat[i,j]] in ['NO', 'MAYBE'] and DESCRIPTOR[llm_mat[i,j]] == 'YES':
            nomaybe_yes_counter[classnames[i]] += 1
            nomaybe_yes_counter[classnames[j]] += 1

    confusions = Counter(['%s ==> %s'%(DESCRIPTOR[gt_mat[ij[0],ij[1]]], DESCRIPTOR[llm_mat[ij[0],ij[1]]]) for ij in diff_ijs])
    print('\nConfusions:')
    print(confusions)
    print('\nNO/MAYBE ==> YES classname counter:')
    print(nomaybe_yes_counter)
    print(len(nomaybe_yes_counter))

def Usage():
    print('python sample_llm_mistakes.py <gt_ternary_filename> <llm_ternary_filename> <num_samples> <random_seed>')


if __name__ == '__main__':
    sample_llm_mistakes(*(sys.argv[1:]))
