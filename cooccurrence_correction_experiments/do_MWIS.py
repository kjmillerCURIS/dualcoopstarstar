import os
import sys
import copy
import numpy as np
import pickle
from scipy.stats import rankdata
import time
import torch
from tqdm import tqdm
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT
from compute_mAP import average_precision
from compute_joint_marg_disagreement import compute_joint_marg_disagreement


DEBUG = False
NUM_CLASSES_DICT = {'COCO2014_partial' : 80}
TOP_K_DICT = {'COCO2014_partial' : 20}
BATCH_SIZE = 64
RESULTS_FILENAME_PREFIX_DICT = {dataset_name : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/MWIS_results/%s-MWIS'%(dataset_name.split('_')[0])) for dataset_name in ['COCO2014_partial']}
SMALL_TYPE_NPY = {False : 'float16', True : 'int16'}
SMALL_TYPE_TORCH = {False : torch.float16, True : torch.int16}


#how to do it:
#-for a single example:
#--sort the score vector and take the TOP_K, and keep the indices
#--use the indices to sort the conflict matrix and keep the TOP_K x TOP_K
#--grab the (constant) powerset matrix (2^TOP_K x 1 x TOP_K) and outer-product it with conflict matrix (1 x TOP_K x TOP_K). This should give you (2^TOP_K x 1 x 1) which you can squeeze to (2^TOP_K,). Binarize that.
#--now multiply powerset matrix by score vector. This gets you vector of total weights (2^TOP_K,).
#--Add together (with extremely high weight for conflicts) and do argmax to select best row of powerset matrix.
#--Add -1's to that row (incase we want to treat 'em differently), and unsort using the indices.
#--Return the unsorted row.
#
#-for a batch of B examples:
#--sort the score vectors (B x TOP_K)
#--sort the conflict matrix (B x TOP_K x TOP_K)
#--get total weights ((B x TOP_K) @ (2^TOP_K x TOP_K).T ==> (B x 2^TOP_K))
#--get infeasibility using einsum, with Q[r,s] = sum_u sum_v P[s,u] C[r,u,v] P[s,v], where P is for powerset and C is for conflict
#--now you can add the two things and argmax across dim=1
#--then use the argmaxes to index rows from powerset matrix (B x TOP_K)
#--pad with -1's and unsort


def make_powerset_matrix(dataset_name):
    print('if i had a million subsets...')
    top_k = TOP_K_DICT[dataset_name]
    powerset_matrix = np.array([[(i >> j) & 1 for j in range(top_k)][::-1] for i in range(2 ** top_k)])
    powerset_matrix = torch.tensor(powerset_matrix, dtype=SMALL_TYPE_TORCH[DEBUG])
    print('i could almost buy a subset of a shoebox')
    return powerset_matrix


#scores should be (B x NUM_CLASSES) and float32
#conflict_matrix should be (NUM_CLASSES x NUM_CLASSES) and float16 and be binary and have zero diagonal
#powerest_matrix should be (2^TOP_K x TOP_K) and float16 and binary
#will return (B x NUM_CLASSES) which will be ternary, with 0 telling you what was eliminated and -1 telling you what was truncated
def MWIS_one_batch(scores, conflict_matrix, powerset_matrix, dataset_name):
    num_classes = NUM_CLASSES_DICT[dataset_name]
    top_k = TOP_K_DICT[dataset_name]
    B = scores.shape[0]
    assert(scores.shape == (B, num_classes))
    assert(scores.dtype == torch.float32)
    assert(conflict_matrix.shape == (num_classes, num_classes))
    assert(conflict_matrix.dtype == SMALL_TYPE_TORCH[DEBUG])
    assert(powerset_matrix.shape == (2 ** top_k, top_k))
    assert(powerset_matrix.dtype == SMALL_TYPE_TORCH[DEBUG])
    with torch.no_grad():

        #sorting and truncating
        sorted_scores, sorting_indices = torch.sort(scores, dim=1, descending=True)
        top_scores = sorted_scores[:,:top_k]
        assert(top_scores.shape == (B, top_k))
        rep_conflict_matrix = conflict_matrix.repeat(B, 1, 1)
        assert(rep_conflict_matrix.shape == (B, num_classes, num_classes))
        top_conflict_matrix = torch.gather(rep_conflict_matrix, 1, torch.unsqueeze(sorting_indices[:,:top_k], dim=2).repeat(1,1,num_classes)) #sort rows, keeping only the best top_k of them
        assert(top_conflict_matrix.shape == (B, top_k, num_classes))
        top_conflict_matrix = torch.gather(top_conflict_matrix, 2, torch.unsqueeze(sorting_indices[:,:top_k], dim=1).repeat(1,top_k,1)) #sort columns, keeping only the best top_k of them
        assert(top_conflict_matrix.shape == (B, top_k, top_k))

        #get total weight of each subset for each example
        total_weights = top_scores @ powerset_matrix.t().float()
        assert(total_weights.shape == (B, 2 ** top_k))

        #get infeasibility of each subset for each example
        infeasibility = torch.einsum('su,ruv,sv->rs', powerset_matrix, top_conflict_matrix, powerset_matrix)
        assert(infeasibility.shape == (B, 2 ** top_k))

        #numerical check on infeasibility
        assert(infeasibility[14,12].item() == torch.squeeze(torch.reshape(powerset_matrix[12,:], (1,-1)) @ top_conflict_matrix[14,:,:] @ torch.reshape(powerset_matrix[12,:], (-1,1))).item())

        #now get the total objectives and maximize them
        objectives = torch.zeros_like(infeasibility, dtype=torch.float32)
        objectives[infeasibility > 0] = float('-inf')
        objectives = objectives + total_weights
        solutions = torch.argmax(objectives, dim=1)
        assert(solutions.shape == (B,))
        assert(torch.all(torch.isfinite(objectives[range(B),solutions])))

        #select subsets and pad
        selections = powerset_matrix[solutions, :]
        assert(selections.shape == (B, top_k))
        if top_k < num_classes:
            selections = torch.cat((selections,-1*torch.ones(size=(B,num_classes-top_k),dtype=selections.dtype,device=selections.device)),dim=1)

        assert(selections.shape == (B, num_classes))

        #unsort the selections
        unsorting_indices = torch.argsort(sorting_indices, dim=1)
        selections = torch.gather(selections, 1, unsorting_indices)
        assert(selections.shape == (B, num_classes))

        #and return!
        return selections


def compute_scores(logits, score_type, gts=None):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        if score_type == 'prob':
            return probs
        elif score_type == 'neglogcompprob':
            return -torch.log(1 - probs)
        elif score_type == 'adaptivelogprob_onemin': #this uses min(probs) to offset logprobs to keep them all above zero
            min_prob = torch.min(probs)
            return torch.log(probs) - torch.log(min_prob)
        elif score_type == 'adaptivelogprob_minperclass': #this computes a separate min per class and uses it to offset the logprobs
            min_prob_per_class = torch.min(probs, 0, keepdim=True)[0]
            return torch.log(probs) - torch.log(min_prob_per_class)
        elif score_type == 'binary_top2perc':
            thresholds = torch.quantile(probs, 0.98, dim=0, keepdim=True)
            return (probs > thresholds).float() + 0.001 * (probs <= thresholds).float()
        elif score_type == 'binary_top5perc':
            thresholds = torch.quantile(probs, 0.95, dim=0, keepdim=True)
            return (probs > thresholds).float() + 0.001 * (probs <= thresholds).float()
        elif score_type == 'binary_top1perc':
            thresholds = torch.quantile(probs, 0.99, dim=0, keepdim=True)
            return (probs > thresholds).float() + 0.001 * (probs <= thresholds).float()
        elif score_type == 'rank_without_gtmargs':
            ranks = rankdata(copy.deepcopy(logits.cpu().numpy()), axis=0)
            pvals = ranks / (logits.shape[0] + 1)
            return torch.tensor(pvals, dtype=logits.dtype, device=logits.device)
        elif score_type == 'rank_with_gtmargs':
            ranks = rankdata(copy.deepcopy(logits.cpu().numpy()), axis=0)
            pvals = ranks / (logits.shape[0] + 1)
            gtmargs = np.mean(np.array([gts[impath] for impath in sorted(gts.keys())]), axis=0, keepdims=True)
            pvals = pvals * gtmargs
            return torch.tensor(pvals, dtype=logits.dtype, device=logits.device)
        else:
            assert(False)


def load_logits_and_gts(dataset_name):
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        logits = pickle.load(f)

    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    if DEBUG:
        impaths = sorted(gts.keys())[:128]
        gts = {impath : gts[impath] for impath in impaths}
        logits = {impath : logits[impath] for impath in impaths}

    return logits, gts


def compute_conflict_matrix(gts, conflict_threshold, dataset_name):
    disagreement = compute_joint_marg_disagreement(gts, dataset_name)
    joint_over_marginals = 1 - disagreement
    conflict_matrix = (joint_over_marginals < conflict_threshold).astype(SMALL_TYPE_NPY[DEBUG])
    conflict_matrix = np.triu(conflict_matrix, k=1)
    conflict_matrix = conflict_matrix + conflict_matrix.T
    assert(np.all(np.diagonal(conflict_matrix) == 0))
    assert(np.all(conflict_matrix == conflict_matrix.T))
    assert(np.all((conflict_matrix == 0) | (conflict_matrix == 1)))
    return torch.tensor(conflict_matrix)


def get_results_filename(dataset_name, conflict_threshold, score_type):
    return RESULTS_FILENAME_PREFIX_DICT[dataset_name] + '-%f-%s.pth.tar'%(conflict_threshold, score_type)


#return mAP_zooNO, mAP_zooYES, APs_zooNO, APs_zooYES
def do_evaluation(logits_orig, selections, gts):
    logits_orig = np.array([logits_orig[impath] for impath in sorted(logits_orig.keys())])
    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    selections = np.array([selections[impath] for impath in sorted(selections.keys())])
    mAP_zoos = []
    APs_zoos = []
    for zoo in [False, True]:
        outputs = copy.deepcopy(logits_orig)
        outputs[selections == 0] = float('-inf')
        if zoo:
            outputs[selections == -1] = float('-inf')

        APs_zoo = []
        for i in range(logits_orig.shape[1]):
            my_AP = average_precision(outputs[:,i], gts[:,i])
            APs_zoo.append(my_AP)

        APs_zoo = np.array(APs_zoo)
        APs_zoos.append(APs_zoo)
        mAP_zoo = np.mean(APs_zoos)
        mAP_zoos.append(mAP_zoo)

    mAP_zooNO, mAP_zooYES = mAP_zoos
    APs_zooNO, APs_zooYES = APs_zoos
    return mAP_zooNO, mAP_zooYES, APs_zooNO, APs_zooYES


def do_MWIS(dataset_name, conflict_threshold, score_type):
    conflict_threshold = float(conflict_threshold)

    logits_orig, gts = load_logits_and_gts(dataset_name)
    logits = torch.tensor(np.array([logits_orig[impath] for impath in sorted(logits_orig.keys())]), dtype=torch.float32)
    conflict_matrix = compute_conflict_matrix(gts, conflict_threshold, dataset_name)
    powerset_matrix = make_powerset_matrix(dataset_name)

    if not DEBUG:
        logits = logits.cuda()
        conflict_matrix = conflict_matrix.cuda()
        powerset_matrix = powerset_matrix.cuda()

    scores = compute_scores(logits, score_type, gts=gts)

    chunk_start = 0
    selections = []
    print('computing..')
    while chunk_start < scores.shape[0]:
        if len(selections) > 0 and len(selections) % 5 == 0:
            print(len(selections))

        chunk_end = min(chunk_start + BATCH_SIZE, scores.shape[0])

        #avoid a batch-size of 1 on the next batch
        if chunk_end == scores.shape[0] - 1:
            chunk_end == scores.shape[0]

        assert(chunk_end - chunk_start > 1)

        scores_batch = scores[chunk_start:chunk_end,:]
        selections_batch = MWIS_one_batch(scores_batch, conflict_matrix, powerset_matrix, dataset_name)
        selections.append(selections_batch)

        chunk_start = chunk_end

    selections = torch.cat(selections, dim=0)
    selections = selections.cpu().numpy()
    selections = {impath : selections[t,:] for t, impath in enumerate(sorted(logits_orig.keys()))}

    mAP_zooNO, mAP_zooYES, APs_zooNO, APs_zooYES = do_evaluation(logits_orig, selections, gts)
    print('mAP_zooNO(%f, %s) = %f'%(conflict_threshold, score_type, mAP_zooNO))
    print('mAP_zooYES(%f, %s) = %f'%(conflict_threshold, score_type, mAP_zooYES))

    results = {'conflict_threshold' : conflict_threshold, 'score_type' : score_type, 'selections' : selections, 'mAP_zooNO' : mAP_zooNO, 'mAP_zooYES' : mAP_zooYES, 'APs_zooNO' : APs_zooNO, 'APs_zooYES' : APs_zooYES}
    results_filename = get_results_filename(dataset_name, conflict_threshold, score_type)
    torch.save(results, results_filename)


def test_with_toy_batch():
    dataset_name = 'COCO2014_partial'
    scores = torch.tensor(np.square(np.random.randn(64, 80)).astype('float32'))
    conflict_matrix = torch.tensor(np.triu((np.random.randn(80,80) > 1.96).astype(SMALL_TYPE_NPY[DEBUG]), k=1))
    with torch.no_grad():
        conflict_matrix = conflict_matrix + conflict_matrix.t()

    powerset_matrix = make_powerset_matrix(dataset_name)
    print('hi')
    start_time = time.time()
    selections = MWIS_one_batch(scores, conflict_matrix, powerset_matrix, dataset_name)
    end_time = time.time()
    print(str(end_time - start_time))
    print('passed!')
    import pdb
    pdb.set_trace()


def usage():
    print('Usage: python do_MWIS.py <dataset_name> <conflict_threshold> <score_type>')


if __name__ == '__main__':
    do_MWIS(*(sys.argv[1:]))
