import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dassl.optim.lr_scheduler import ConstantWarmupScheduler
from harvest_mscoco_training_gts import MSCOCO_TRAINING_GTS_FILENAME
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME
from compute_joint_marg_disagreement import C_FILENAME
from compute_mAP import compute_mAP
from tune_pseudolabels_negpenalty import general_cooccurrence_cost, BATCH_SIZE, NUM_EPOCHS, LR, WARMUP_LR, MOMENTUM, setup_optimizer, setup_scheduler



RESULTS_PREFIX = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training-oracle-negpos-prob_prob_constrained-results/mscoco_training-oracle-negpos-prob_prob_constrained-results')


def total_cost(logits, init_logits, C, params):
    p = params
    init_probs = torch.sigmoid(init_logits)
    bce_cost = torch.sum(torch.sum(F.binary_cross_entropy_with_logits(logits, init_probs, reduction='none'), dim=1))
    negpos_cooc_cost = general_cooccurrence_cost(logits, C, p['negpos_cost_type'])
    return bce_cost + negpos_cooc_cost


#this time we bake alpha and beta into C, so other stuff can just multiply into it
def preprocess_C(C, params):
    p = params
    C_out = np.zeros_like(C)
    C_out[C > p['epsilon']] = p['alpha']
    C_out[C < p['zeta']] = -p['beta']
    C_out = np.triu(C_out, k=1)
    return C_out


#return tuned_logits, results
#assume C is already good-to-go, upper triangular, etc.
#tuned_logits should be in the SAME format as pseudolabel_logits
#no need to compute mAP, that will happen either outside this function or in another script (probably the former)
#(we'll probably make another script just to do the baseline mAP)
def do_tuning(pseudolabel_logits, C, params):
    p = params
    results = {}
    C = torch.tensor(C, device='cuda')
    impaths = sorted(pseudolabel_logits.keys())
    init_logits = torch.tensor([pseudolabel_logits[impath] for impath in impaths], device='cuda')
    intermediate_logits, costs_constrained_phase = do_tuning_helper(init_logits, init_logits, C, p, is_constrained=True)
    tuned_logits, costs_unconstrained_phase = do_tuning_helper(intermediate_logits, init_logits, C, p, is_constrained=False)
    results['costs_constrained_phase'] = costs_constrained_phase
    results['costs_unconstrained_phase'] = costs_unconstrained_phase
    intermediate_logits = intermediate_logits.cpu().numpy()
    tuned_logits = tuned_logits.cpu().numpy()
    intermediate_logits = {impath : v for impath, v in zip(impaths, intermediate_logits)}
    tuned_logits = {impath : v for impath, v in zip(impaths, tuned_logits)}
    results['intermediate_logits'] = intermediate_logits
    results['tuned_logits'] = tuned_logits
    return tuned_logits, results


#returns logits, costs
def do_tuning_helper(init_logits_for_starting_point, init_logits_for_proximal_loss, C, params, is_constrained=False):
    p = params
    logits = nn.parameter.Parameter(init_logits_for_starting_point.detach().clone())
    optimizer = setup_optimizer(logits, p)
    scheduler = setup_scheduler(optimizer, p, custom_num_epochs=p['num_epochs'] // 2)
    costs = []
    for epoch in tqdm(range(p['num_epochs'] // 2)):
        costs_one_epoch = []
        for _ in range(p['num_steps_per_epoch']):
            if is_constrained:
                with torch.no_grad():
                    gamma_for_logit = np.fabs(np.log(1 - p['gamma']) - np.log(p['gamma']))
                    logits.clamp(-gamma_for_logit, gamma_for_logit)

            optimizer.zero_grad()
            cost = total_cost(logits, init_logits_for_proximal_loss, C, p)
            costs_one_epoch.append(cost.item())
            cost.backward()
            optimizer.step()

        costs.append(np.mean(costs_one_epoch))
        print('epoch %d (is_constrained=%s): train_cost=%f'%(epoch, str(is_constrained), costs[-1]))
        scheduler.step()

    return logits.data, costs


def tune_pseudolabels_negpos_prob_prob_constrained(alpha, epsilon, beta, zeta, gamma):
    alpha = float(alpha)
    epsilon = float(epsilon)
    beta = float(beta)
    zeta = float(zeta)
    gamma = float(gamma)

    p = {'alpha' : alpha, 'epsilon' : epsilon, 'beta' : beta, 'zeta' : zeta, 'gamma' : gamma, 'negpos_cost_type' : 'prob_prob'}
    p['num_steps_per_epoch'] = int(round(80000 / BATCH_SIZE))
    p['num_epochs'] = NUM_EPOCHS
    p['optimizer'] = 'sgdA'
    p['scheduler'] = 'cosineA'
    with open(C_FILENAME, 'rb') as f:
        C = pickle.load(f)

    C = preprocess_C(C, p)
    with open(PSEUDOLABEL_LOGITS_FILENAME, 'rb') as f:
        pseudolabel_logits = pickle.load(f)

    #get this part BEFORE training so I don't do an entire training just to watch it all crash!
    with open(MSCOCO_TRAINING_GTS_FILENAME, 'rb') as f:
        gt_labels = pickle.load(f)

    tuned_logits, results = do_tuning(pseudolabel_logits, C, p)
    results['params'] = p
    mAP = compute_mAP(tuned_logits, gt_labels)
    results['mAP'] = mAP

    results_filename = RESULTS_PREFIX + '-alpha%f-epsilon%f-beta%f-zeta%f-gamma%f.pkl'%(alpha, epsilon, beta, zeta, gamma)
    if not os.path.exists(os.path.dirname(results_filename)):
        os.makedirs(os.path.dirname(results_filename))

    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

    print('mAP(%f, %f, %f, %f, %s) = %f'%(alpha, epsilon, beta, zeta, gamma, mAP))


def usage():
    print('Usage: python tune_pseudolabels_negpos_prob_prob_constrained.py <alpha> <epsilon> <beta> <zeta> <gamma>')


if __name__ == '__main__':
    tune_pseudolabels_negpos_prob_prob_constrained(*(sys.argv[1:]))
