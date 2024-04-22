import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dassl.optim.lr_scheduler import ConstantWarmupScheduler


NUM_EPOCHS = 1000
NUM_STEPS_PER_EPOCH = 20 #in the original implementation I thought I was taking batches but was actually updating the whole set of pseudolabels at every step
LR = 5e-3
WARMUP_LR = 1e-5
MOMENTUM = 0.0 #no point in momentum when 95% of the grad will be 0, and it'll always be a different 5%


#we get a matrix of pearson correlations, negate it, multiply by C, and sum (*not* mean) over the dimensions of C
#if put_on_sum_scale is True, then we multiply it by logits.shape[0], else we return it as-is
#you might want to leave put_on_sum_scale as True for pseudolabel correction, since the BCE part uses a sum
#this also gets used as a helper by correlation_cooccurrence_cost_learnable_sigmoid(), in which case "logits" is actually the output of that sigmoid
def correlation_cooccurrence_cost(logits, ternary_cooccurrence_mat, put_on_sum_scale=True):
    logits = torch.unsqueeze(logits, dim=-1) #(B, N, 1)
    means = torch.mean(logits, dim=0) #(N, 1)
    meansquares = torch.mean(torch.square(logits), dim=0) #(N, 1)
    sds = torch.sqrt(meansquares - torch.square(means)) #(N, 1)
    meandots = torch.mean(torch.bmm(logits, torch.permute(logits, (0,2,1))), dim=0) #(N, N)
    numerators = meandots - means @ means.t() #(N, N)
    denominators = sds @ sds.t() #(N, N)
    correlations = numerators / torch.clip(denominators, min=1e-6)
    cost = torch.sum(ternary_cooccurrence_mat * correlations) #yes, you always do a *sum* here! and no, you should *not* make this negative, the current signage is already correct!
    if put_on_sum_scale:
        cost = logits.shape[0] * cost

    return cost


#assume that C is upper-triangle
#use sum, because it's just a bunch of independent optimization problems
def general_cooccurrence_cost(logits, ternary_cooccurrence_mat, cost_type):
    assert(len(logits.shape) == 2)
    assert(ternary_cooccurrence_mat.shape == (logits.shape[1], logits.shape[1]))
    if cost_type == 'correlation':
        return correlation_cooccurrence_cost(logits, ternary_cooccurrence_mat)

    assert(cost_type != 'correlation')
    logits = torch.unsqueeze(logits, dim=-1) #(B, N, 1)
    probs = torch.sigmoid(logits)
    if cost_type == 'prob_prob':
        product = probs @ torch.permute(probs, (0,2,1))
    elif cost_type == 'prob_logit':
        product = 0.5 * probs @ torch.permute(logits, (0,2,1)) + 0.5 * logits @ torch.permute(probs, (0,2,1))
    elif cost_type == 'prob_stopgrad_logit':
        probs = probs.detach()
        product = 0.5 * probs @ torch.permute(logits, (0,2,1)) + 0.5 * logits @ torch.permute(probs, (0,2,1))
    else:
        assert(False)

    assert(product.shape == (logits.shape[0], ternary_cooccurrence_mat.shape[0], ternary_cooccurrence_mat.shape[1]))
    cost = torch.sum(torch.unsqueeze(ternary_cooccurrence_mat, dim=0) * product, dim=(1,2))
    cost = torch.sum(cost)
    return cost


def total_cost(logits, init_logits, ternary_cooccurrence_mat, cost_type):
    init_probs = torch.sigmoid(init_logits)
    bce_cost = torch.sum(torch.sum(F.binary_cross_entropy_with_logits(logits, init_probs, reduction='none'), dim=1))
    negpos_cooc_cost = general_cooccurrence_cost(logits, ternary_cooccurrence_mat, cost_type)
    return bce_cost + negpos_cooc_cost


#return tuned_logits
#assume ternary_cooccurrence_mat is already good-to-go, upper triangular, etc.
#tuned_logits should be in the SAME format as pseudolabel_logits
def correct_initial_pseudolabels(pseudolabel_logits, ternary_cooccurrence_mat, cost_type):
    init_logits = pseudolabel_logits.detach().clone()
    logits = nn.parameter.Parameter(init_logits.detach().clone())
    optimizer = setup_optimizer(logits)
    scheduler = setup_scheduler(optimizer)
    for epoch in tqdm(range(NUM_EPOCHS)):
        costs = []
        for _ in tqdm(range(NUM_STEPS_PER_EPOCH)):
            optimizer.zero_grad()
            cost = total_cost(logits, init_logits, ternary_cooccurrence_mat, cost_type)
            costs.append(cost.item() / len(logits))
            cost.backward()
            optimizer.step()

        print('epoch %d: train_cost=%f'%(epoch, np.mean(costs)))
        scheduler.step()

    tuned_logits = logits.data.detach().clone()
    return tuned_logits


def setup_optimizer(logits):
    optimizer = torch.optim.SGD([logits], lr=LR, momentum=MOMENTUM) #same as Dassl, but no weight decay because we already have proximal regularization
    return optimizer


def setup_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    scheduler = ConstantWarmupScheduler(optimizer, scheduler, 1, WARMUP_LR)
    return scheduler
