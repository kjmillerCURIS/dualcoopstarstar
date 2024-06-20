import os
import sys
import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from dassl.optim.lr_scheduler import ConstantWarmupScheduler
from compute_mAP import average_precision


METRIC_DENOMINATOR_FLOOR = 1e-6
BCE_FLOOR = 1e-8 #just for BCE. inverse sigmoid will return +/-inf if given 0 or 1, which I'm okay with
#rationale for 1e-8 is that things below it might get squished to 0, at which point their ordering would be destroyed...so I'm basically trading off that squishing vs possibility of large gradients
OPTIMIZER = 'sgdA'
SCHEDULER = 'cosineA'
NUM_EPOCHS = 10000
NUM_STEPS_PER_EPOCH = 20
LR = 2e-6 #5e-3
WARMUP_LR = 1e-8 #1e-5
MOMENTUM = 0.0
OUTPUT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/tune_pseudolabels_metric_matching_results')


#pseudolabel_probs should be shape (N, num_classes) and should be in [0,1] range
#return d_hat_metric, which has shape (num_classes, num_classes) and is differentiable
def compute_d_hat_metric(pseudolabel_probs):
    assert(len(pseudolabel_probs.shape) == 2)
    assert(torch.all(torch.logical_and(pseudolabel_probs >= 0, pseudolabel_probs <= 1)))
    joints = pseudolabel_probs.T @ pseudolabel_probs / pseudolabel_probs.shape[0]
    margs = torch.mean(pseudolabel_probs, dim=0, keepdim=True)
    indeps = margs.T @ margs
    indeps = indeps + (torch.clamp(indeps, min=METRIC_DENOMINATOR_FLOOR) - indeps).detach()
    return joints / indeps


#actually gives H(P,Q) - H(P) (i.e. KL-divergence) instead of just H(P,Q), but H(P,Q) is the only differentiable part
#this is done so that it will be zero if the probabilities match perfectly
def compute_proximal_BCE_loss(pseudolabel_probs, init_pseudolabel_probs):
    assert(len(pseudolabel_probs.shape) == 2)
    assert(len(init_pseudolabel_probs.shape) == 2)
    assert(torch.all(torch.logical_and(pseudolabel_probs >= 0, pseudolabel_probs <= 1)))
    assert(torch.all(torch.logical_and(init_pseudolabel_probs >= 0, init_pseudolabel_probs <= 1)))
    pseudolabel_probs = pseudolabel_probs + (torch.clamp(pseudolabel_probs, min=BCE_FLOOR, max=1-BCE_FLOOR) - pseudolabel_probs).detach()
    with torch.no_grad():
        entropy_part = torch.sum(torch.sum(-(init_pseudolabel_probs * torch.log(init_pseudolabel_probs) + (1 - init_pseudolabel_probs) * torch.log(1 - init_pseudolabel_probs))))

    return torch.sum(torch.sum(F.binary_cross_entropy(pseudolabel_probs, init_pseudolabel_probs, reduction='none'), dim=1)) - entropy_part


#yes, this DOES include the alpha part (in case we try something fancy with it later)
#d_metric should be size (num_classes, num_classes)
def compute_metric_loss(pseudolabel_probs, d_metric, alpha):
    d_hat_metric = compute_d_hat_metric(pseudolabel_probs)
    diffs = torch.triu(d_hat_metric - d_metric, diagonal=1)
    return alpha * torch.sum(torch.square(diffs))


def compute_total_loss(pseudolabel_probs, d_metric, alpha, init_pseudolabel_probs):
    metric_loss = compute_metric_loss(pseudolabel_probs, d_metric, alpha)
    proximal_BCE_loss = compute_proximal_BCE_loss(pseudolabel_probs, init_pseudolabel_probs)
    total_loss = metric_loss + proximal_BCE_loss
    return total_loss, proximal_BCE_loss, metric_loss


#note that this does NOT do any steps on the scheduler!
def one_optimization_step(pseudolabel_probs, d_metric, alpha, init_pseudolabel_probs, optimizer):
    optimizer.zero_grad()
    loss, proximal_loss, metric_loss = compute_total_loss(pseudolabel_probs, d_metric, alpha, init_pseudolabel_probs)
    loss.backward()
    gradnorm = pseudolabel_probs.grad.norm().item()
    gradmax = torch.max(torch.abs(pseudolabel_probs.grad)).item()
    optimizer.step()
    pseudolabel_probs.data.clamp_(0,1)
    return loss.item(), proximal_loss.item(), metric_loss.item(), gradnorm, gradmax


#return as 2D arrays
def load_gts_and_pseudolabel_logits(dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        logits = pickle.load(f)

    logits = np.array([logits[impath] for impath in sorted(logits.keys())])
    return gts, logits


#take in gts as 2D array
#return (num_classes, num_classes) array (someone else will have to turn it into tensor)
def form_d_metric(gts):
    joints = gts.T @ gts / gts.shape[0]
    margs = np.mean(gts, axis=0, keepdims=True)
    indeps = margs.T @ margs
    indeps = np.maximum(indeps, METRIC_DENOMINATOR_FLOOR)
    d_metric = joints / indeps
    return d_metric


def setup_optimizer(pseudolabel_probs):
    assert(OPTIMIZER == 'sgdA')
    optimizer = torch.optim.SGD([pseudolabel_probs], lr=LR, momentum=MOMENTUM) #same as Dassl, but no weight decay because we already have proximal regularization
    return optimizer


def setup_scheduler(optimizer):
    assert(SCHEDULER == 'cosineA')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    scheduler = ConstantWarmupScheduler(optimizer, scheduler, 1, WARMUP_LR)
    return scheduler


def make_results_dict(mAP, tuned_pseudolabel_probs, losses, proximal_losses, metric_losses, gradnorms, gradmaxes, init_metric_loss, alpha, dataset_name):
    params = {'alpha' : alpha, 'metric_denominator_floor' : METRIC_DENOMINATOR_FLOOR, 'bce_floor' : BCE_FLOOR, 'optimizer' : OPTIMIZER, 'scheduler' : SCHEDULER, 'num_epochs' : NUM_EPOCHS, 'num_steps_per_epoch' : NUM_STEPS_PER_EPOCH, 'lr' : LR, 'warmup_lr' : WARMUP_LR, 'momentum' : MOMENTUM}
    results_dict = {'mAP' : mAP, 'tuned_pseudolabel_probs' : tuned_pseudolabel_probs, 'dataset_name' : dataset_name, 'losses' : losses, 'proximal_losses' : proximal_losses, 'metric_losses' : metric_losses, 'gradnorms' : gradnorms, 'gradmaxes' : gradmaxes, 'init_metric_loss' : init_metric_loss, 'params' : params}
    return results_dict


def get_results_dict_filename(dataset_name, alpha):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, 'tune_pseudolabels_metric_matching_%s_alpha%s_bce_floor%s_lr%s_warmup_lr%s_num_epochs%d.pth'%(dataset_name.split('_')[0], str(alpha), str(BCE_FLOOR), str(LR), str(WARMUP_LR), NUM_EPOCHS))


def compute_mAP(tuned_pseudolabel_probs, gts):
    return np.mean([average_precision(tuned_pseudolabel_probs[:,i], gts[:,i]) for i in range(gts.shape[1])])


def tune_pseudolabels_metric_matching(dataset_name, alpha):
    alpha = float(alpha)

    gts, pseudolabel_logits = load_gts_and_pseudolabel_logits(dataset_name)
    d_metric = form_d_metric(gts)
    d_metric = torch.tensor(d_metric, device='cuda')
    pseudolabel_logits = torch.tensor(pseudolabel_logits, device='cuda')
    with torch.no_grad():
        source_pseudolabel_probs = torch.sigmoid(pseudolabel_logits)

    pseudolabel_probs = nn.parameter.Parameter(source_pseudolabel_probs.detach().clone())
    init_pseudolabel_probs = source_pseudolabel_probs.detach().clone()
    with torch.no_grad():
        init_metric_loss = compute_metric_loss(init_pseudolabel_probs, d_metric, alpha)
        init_metric_loss = init_metric_loss.item()

    optimizer = setup_optimizer(pseudolabel_probs)
    scheduler = setup_scheduler(optimizer)
    losses = []
    proximal_losses = []
    metric_losses = []
    gradnorms = []
    gradmaxes = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        losses_one_epoch = []
        proximal_losses_one_epoch = []
        metric_losses_one_epoch = []
        gradnorms_one_epoch = []
        gradmaxes_one_epoch = []
        for _ in range(NUM_STEPS_PER_EPOCH):
            loss, proximal_loss, metric_loss, gradnorm, gradmax = one_optimization_step(pseudolabel_probs, d_metric, alpha, init_pseudolabel_probs, optimizer)
            losses_one_epoch.append(loss)
            proximal_losses_one_epoch.append(proximal_loss)
            metric_losses_one_epoch.append(metric_loss)
            gradnorms_one_epoch.append(gradnorm)
            gradmaxes_one_epoch.append(gradmax)

        losses.append(losses_one_epoch)
        proximal_losses.append(proximal_losses_one_epoch)
        metric_losses.append(metric_losses_one_epoch)
        gradnorms.append(gradnorms_one_epoch)
        gradmaxes.append(gradmaxes_one_epoch)
        print('epoch %d: loss=%f, proximal_loss=%f, metric_loss=%f, gradnorm=%f, gradmax=%f, init_metric_loss=%f'%(epoch, np.mean(losses[-1]), np.mean(proximal_losses[-1]), np.mean(metric_losses[-1]), np.mean(gradnorms[-1]), np.mean(gradmaxes[-1]), init_metric_loss))
        scheduler.step()

    tuned_pseudolabel_probs = pseudolabel_probs.data.detach().cpu().numpy()
    mAP = compute_mAP(tuned_pseudolabel_probs, gts)
    print('mAP(alpha=%s, bce_floor=%s, lr=%s, warmup_lr=%s, num_epochs=%d) = %f'%(str(alpha), str(BCE_FLOOR), str(LR), str(WARMUP_LR), NUM_EPOCHS, mAP))

    results_dict = make_results_dict(mAP, tuned_pseudolabel_probs, losses, proximal_losses, metric_losses, gradnorms, gradmaxes, init_metric_loss, alpha, dataset_name)
    results_dict_filename = get_results_dict_filename(dataset_name, alpha)
    torch.save(results_dict, results_dict_filename)


def usage():
    print('Usage: python tune_pseudolabels_metric_matching.py <dataset_name> <alpha>')


if __name__ == '__main__':
    tune_pseudolabels_metric_matching(*(sys.argv[1:]))
