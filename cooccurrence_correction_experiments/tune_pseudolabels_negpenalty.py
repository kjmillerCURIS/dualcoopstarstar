import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dassl.optim.lr_scheduler import ConstantWarmupScheduler
from harvest_mscoco_training_gts import MSCOCO_TRAINING_GTS_FILENAME
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME
from compute_joint_marg_disagreement import C_FILENAME
from compute_mAP import compute_mAP


BATCH_SIZE = 4096
NUM_EPOCHS = 1000
LR = 5e-3
WARMUP_LR = 1e-5
MOMENTUM = 0.0 #no point in momentum when 95% of the grad will be 0, and it'll always be a different 5%
RESULTS_PREFIX = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training-oracle-negpenalty-results/mscoco_training-oracle-negpenalty-results')


class IndexDataset(Dataset):
    def __init__(self, num_examples):
        self.num_examples = num_examples

    def __getitem__(self, idx):
        return {'idx' : idx}

    def __len__(self):
        return self.num_examples


#will have to multiply alpha into it outside of this
def negative_cooccurrence_cost(logits, C, params):
    p = params
    return general_cooccurrence_cost(logits, C, p['neg_cost_type'])


#we get a matrix of pearson correlations, negate it, multiply by C, and sum (*not* mean) over the dimensions of C
#if put_on_sum_scale is True, then we multiply it by logits.shape[0], else we return it as-is
#you might want to leave put_on_sum_scale as True for pseudolabel correction, since the BCE part uses a sum
#this also gets used as a helper by correlation_cooccurrence_cost_learnable_sigmoid(), in which case "logits" is actually the output of that sigmoid
def correlation_cooccurrence_cost(logits, C, put_on_sum_scale=True):
    logits = torch.unsqueeze(logits, dim=-1) #(B, N, 1)
    means = torch.mean(logits, dim=0) #(N, 1)
    meansquares = torch.mean(torch.square(logits), dim=0) #(N, 1)
    sds = torch.sqrt(meansquares - torch.square(means)) #(N, 1)
    meandots = torch.mean(torch.bmm(logits, torch.permute(logits, (0,2,1))), dim=0) #(N, N)
    numerators = meandots - means @ means.t() #(N, N)
    denominators = sds @ sds.t() #(N, N)
    correlations = numerators / torch.clip(denominators, min=1e-6)
    cost = torch.sum(C * correlations) #yes, you always do a *sum* here! and no, you should *not* make this negative, the current signage is already correct!
    if put_on_sum_scale:
        cost = logits.shape[0] * cost

    return cost


def correlation_cooccurrence_cost_learnable_sigmoid(logits, C, extra_learnables, put_on_sum_scale=True):
    logit_scale, logit_bias = extra_learnables['logit_scale'], extra_learnables['logit_bias']
    logit_scale, logit_bias = torch.unsqueeze(logit_scale, dim=0), torch.unsqueeze(logit_bias, dim=0)
    probs = torch.sigmoid(logit_scale.exp() * (logits - logit_bias))
    return correlation_cooccurrence_cost(probs, C, put_on_sum_scale=put_on_sum_scale)


#assume that C is upper-triangle
#use sum, because it's just a bunch of independent optimization problems
def general_cooccurrence_cost(logits, C, cost_type, extra_learnables=None, params=None):
    p = params
    assert(len(logits.shape) == 2)
    assert(C.shape == (logits.shape[1], logits.shape[1]))
    if cost_type == 'correlation':
        return correlation_cooccurrence_cost(logits, C)
    elif cost_type == 'correlation_learnable_sigmoid':
        assert(extra_learnables is not None)
        return correlation_cooccurrence_cost_learnable_sigmoid(logits, C, extra_learnables)

    assert(cost_type not in ['correlation', 'correlation_learnable_sigmoid'])
    logits = torch.unsqueeze(logits, dim=-1) #(B, N, 1)
    probs = torch.sigmoid(logits)
    if cost_type == 'prob_prob':
        product = probs @ torch.permute(probs, (0,2,1))
    elif cost_type == 'prob_logit':
        product = 0.5 * probs @ torch.permute(logits, (0,2,1)) + 0.5 * logits @ torch.permute(probs, (0,2,1))
    elif cost_type == 'prob_stopgrad_logit':
        probs = probs.detach()
        product = 0.5 * probs @ torch.permute(logits, (0,2,1)) + 0.5 * logits @ torch.permute(probs, (0,2,1))
    elif cost_type == 'prob_stopgrad_dampedlogit':
        assert(p is not None)
        probs = probs.detach()
        dampedlogits = 4 * p['rho'] * (torch.sigmoid(logits / p['rho']) - 0.5)
        product = 0.5 * probs @ torch.permute(dampedlogits, (0,2,1)) + 0.5 * dampedlogits @ torch.permute(probs, (0,2,1))
    elif cost_type == 'prob_stopgrad_neglog':
        #defining this in a way that makes it play nicely with a signed C
        #first, define the prob*neglog product for cooccurrences that should be rewarded and penalized
        #then, select penalty, and *negative* reward, and multiply into C
        probs = probs.detach()
        neglog_reward = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
        neglog_penalize = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
        product_reward = 0.5 * probs @ torch.permute(neglog_reward, (0,2,1)) + 0.5 * neglog_reward @ torch.permute(probs, (0,2,1))
        product_penalize = 0.5 * probs @ torch.permute(neglog_penalize, (0,2,1)) + 0.5 * neglog_penalize @ torch.permute(probs, (0,2,1))
        with torch.no_grad():
            reward_selector = (C < 0).float()
            penalize_selector = (C > 0).float()

        product = penalize_selector * product_penalize - reward_selector * product_reward #subtract reward part to cancel out the -beta
    else:
        assert(False)

    assert(product.shape == (logits.shape[0], C.shape[0], C.shape[1]))
    cost = torch.sum(torch.unsqueeze(C, dim=0) * product, dim=(1,2))
    cost = torch.sum(cost)
    return cost


def total_cost(logits, init_logits, C, params):
    p = params
    init_probs = torch.sigmoid(init_logits)
    bce_cost = torch.sum(torch.sum(F.binary_cross_entropy_with_logits(logits, init_probs, reduction='none'), dim=1))
    neg_cooc_cost = negative_cooccurrence_cost(logits, C, p)
    return bce_cost + p['alpha'] * neg_cooc_cost


def setup_optimizer(wlist, params):
    p = params
    assert(p['optimizer'] == 'sgdA')
    if not isinstance(wlist, list):
        wlist = [wlist]

    optimizer = torch.optim.SGD(wlist, lr=LR, momentum=MOMENTUM) #same as Dassl, but no weight decay because we already have proximal regularization
    return optimizer


def setup_scheduler(optimizer, params, custom_num_epochs=None):
    p = params
    assert(p['scheduler'] == 'cosineA')
    if custom_num_epochs is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, p['num_epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, custom_num_epochs)
    
    scheduler = ConstantWarmupScheduler(optimizer, scheduler, 1, WARMUP_LR)
    return scheduler


def preprocess_C(C, params):
    p = params
    C = (C > p['epsilon']).astype('float32')
    C = np.triu(C, k=1)
    return C


#return tuned_logits, results
#assume C is already good-to-go, upper triangular, etc.
#tuned_logits should be in the SAME format as pseudolabel_logits
#no need to compute mAP, that will happen either outside this function or in another script (probably the former)
#(we'll probably make another script just to do the baseline mAP)
def do_tuning(pseudolabel_logits, C, params):
    p = params
    results = {}
    index_dataset = IndexDataset(len(pseudolabel_logits))
    index_dataloader = DataLoader(index_dataset, batch_size=p['batch_size'], shuffle=True)
    C = torch.tensor(C, device='cuda')
    impaths = sorted(pseudolabel_logits.keys())
    init_logits = torch.tensor([pseudolabel_logits[impath] for impath in impaths], device='cuda')
    logits = nn.parameter.Parameter(init_logits.detach().clone())
    optimizer = setup_optimizer(logits, p)
    scheduler = setup_scheduler(optimizer, p)
    results['costs'] = []
    for epoch in tqdm(range(p['num_epochs'])):
        cost_accumulator = 0.0
        cost_counter = 0.0
        for index_batch in tqdm(index_dataloader):
            optimizer.zero_grad()
            logits_batch = logits[index_batch['idx']]
            cost = total_cost(logits, init_logits, C, p)
            cost_accumulator += cost.item()
            cost_counter += index_batch['idx'].shape[0]
            cost.backward()
            optimizer.step()

        results['costs'].append(cost_accumulator / cost_counter)
        print('epoch %d: train_cost=%f'%(epoch, results['costs'][-1]))
        scheduler.step()

    tuned_logits = logits.data.cpu().numpy()
    tuned_logits = {impath : v for impath, v in zip(impaths, tuned_logits)}
    results['tuned_logits'] = tuned_logits
    return tuned_logits, results


def tune_pseudolabels_negpenalty(alpha, epsilon, neg_cost_type):
    alpha = float(alpha)
    epsilon = float(epsilon)

    p = {'alpha' : alpha, 'epsilon' : epsilon, 'neg_cost_type' : neg_cost_type}
    p['batch_size'] = BATCH_SIZE
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

    results_filename = RESULTS_PREFIX + '-alpha%f-epsilon%f-neg_cost_type%s.pkl'%(alpha, epsilon, neg_cost_type)
    if not os.path.exists(os.path.dirname(results_filename)):
        os.makedirs(os.path.dirname(results_filename))

    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

    print('mAP(%f, %f, %s) = %f'%(alpha, epsilon, neg_cost_type, mAP))


def usage():
    print('Usage: python tune_pseudolabels_negpenalty.py <alpha> <epsilon> <neg_cost_type>')


if __name__ == '__main__':
    tune_pseudolabels_negpenalty(*(sys.argv[1:]))
