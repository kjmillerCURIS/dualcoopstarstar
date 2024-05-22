import os
import sys
import numpy as np
import pickle
import pprint
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT, get_data_manager
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT


#return gts, logits, classnames as 2D arrays
def load_data(dataset_name):
    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
        gts = pickle.load(f)

    gts = np.array([gts[impath] for impath in sorted(gts.keys())])
    with open(PSEUDOLABEL_LOGITS_FILENAME_DICT[dataset_name], 'rb') as f:
        logits = pickle.load(f)

    logits = np.array([logits[impath] for impath in sorted(logits.keys())])
    dm = get_data_manager(dataset_name)
    classnames = dm.dataset.classnames
    return gts, logits, classnames


#return margs[i] = np.mean(scores[:,i])
def compute_margs_from_scores(scores):
    return np.mean(1.0 * scores, axis=0)


#return joint[i,j] = np.mean(scores[:,i] * scores[:,j])
def compute_joint_from_scores(scores):
    joint = scores.T @ scores / (1.0 * scores.shape[0])
    assert(joint.shape == (scores.shape[1], scores.shape[1]))
    return joint


#return d_metric[i,j] = np.mean(scores[:,i] * scores[:,j]) / (np.mean(scores[:,i]) * np.mean(scores[:,j]))
def compute_metric_from_scores(scores):
    margs = compute_margs_from_scores(scores)
    joint = compute_joint_from_scores(scores)
    return joint / (margs[:, np.newaxis] * margs[np.newaxis, :])


#return lambda_vec[i] = E[logits[:,i] | gts[:,i] = 0]
def compute_lambda_vec(gts, probs):
    return np.sum((1. - gts) * probs, axis=0) / np.sum(1. - gts, axis=0)


#return lambda_vec[i] = E[logits[:,i] | gts[:,i] = 1]
def compute_mu_vec(gts, probs):
    return np.sum(gts * probs, axis=0) / np.sum(1.0 * gts, axis=0)


def compute_r(gts, probs, mu_vec, lambda_vec):
    r = probs - (gts * mu_vec + (1 - gts) * lambda_vec)
    assert(np.all(np.fabs(np.sum(gts * r, axis=0) / np.sum(1.0 * gts)) < 1e-6))
    assert(np.all(np.fabs(np.sum((1. - gts) * r, axis=0) / np.sum(1. - gts)) < 1e-6))
    return r


def compute_b(gts, mu_vec, lambda_vec):
    lambda_rel = lambda_vec / (mu_vec - lambda_vec)
    margs = compute_margs_from_scores(gts)
    b = margs[np.newaxis,:] * lambda_rel[:,np.newaxis]
    b = b + b.T + lambda_rel[np.newaxis,:] * lambda_rel[:,np.newaxis]
    return b


def compute_q(gts, mu_vec, lambda_vec, r):
    assert(np.all(np.fabs(np.mean(r, axis=0)) < 1e-6))
    covrr = r.T @ r / r.shape[0] #this assumes that r is zero-mean
    Egr = gts.T @ r / r.shape[0]
    q = Egr / (mu_vec - lambda_vec)[np.newaxis,:] #match denominator to index of r
    q = q + q.T + covrr / ((mu_vec - lambda_vec)[np.newaxis,:] * (mu_vec - lambda_vec)[:,np.newaxis])
    return q


def compute_d_hat_metric_sanity(gts, b, q):
    joint = compute_joint_from_scores(gts)
    margs = compute_margs_from_scores(gts)
    return (joint + b + q) / (margs[:,np.newaxis] * margs[np.newaxis,:] + b)


def compute_d_tilde_metric_sanity(gts, q):
    joint = compute_joint_from_scores(gts)
    margs = compute_margs_from_scores(gts)
    return (joint + q) / (margs[:,np.newaxis] * margs[np.newaxis,:])


def triu_flatten(A):
    assert(len(A.shape) == 2)
    assert(A.shape[0] == A.shape[1])
    row_indices, col_indices = np.triu_indices(A.shape[0], k=1)
    return A[row_indices, col_indices]


#return stats_dict
def compute_stats(d_metric, d_hat_metric, d_tilde_metric, d_star_metric, b, q, d_hat_metric_sanity, d_tilde_metric_sanity):
    d_metric = triu_flatten(d_metric)
    d_hat_metric = triu_flatten(d_hat_metric)
    d_tilde_metric = triu_flatten(d_tilde_metric)
    d_star_metric = triu_flatten(d_star_metric)
    b = triu_flatten(b)
    q = triu_flatten(q)
    d_hat_metric_sanity = triu_flatten(d_hat_metric_sanity)
    d_tilde_metric_sanity = triu_flatten(d_tilde_metric_sanity)
    assert(np.all(np.fabs(d_hat_metric_sanity - d_hat_metric) < 1e-2))
    assert(np.all(np.fabs(d_hat_metric_sanity - d_hat_metric)[np.fabs(d_hat_metric) < 0.1] < 1e-4))
    assert(np.all(np.fabs(d_tilde_metric_sanity - d_tilde_metric) < 1e-2))
    assert(np.all(np.fabs(d_tilde_metric_sanity - d_tilde_metric)[np.fabs(d_tilde_metric) < 0.1] < 1e-4))
    stats_dict = {}
    stats_dict['RMSD(d_hat, d)'] = np.sqrt(np.mean(np.square(d_hat_metric, d_metric)))
    stats_dict['RMSD(d_tilde, d)'] = np.sqrt(np.mean(np.square(d_tilde_metric, d_metric)))
    stats_dict['RMSD(d_star, d)'] = np.sqrt(np.mean(np.square(d_star_metric, d_metric)))
    stats_dict['RMSDclamp01(d_hat, d)'] = np.sqrt(np.mean(np.square(np.clip(d_hat_metric, 0, 1), np.clip(d_metric, 0, 1))))
    stats_dict['RMSDclamp01(d_tilde, d)'] = np.sqrt(np.mean(np.square(np.clip(d_tilde_metric, 0, 1), np.clip(d_metric, 0, 1))))
    stats_dict['RMSDclamp01(d_star, d)'] = np.sqrt(np.mean(np.square(np.clip(d_star_metric, 0, 1), np.clip(d_metric, 0, 1))))
    stats_dict['RMSDclamp02(d_hat, d)'] = np.sqrt(np.mean(np.square(np.clip(d_hat_metric, 0, 2), np.clip(d_metric, 0, 2))))
    stats_dict['RMSDclamp02(d_tilde, d)'] = np.sqrt(np.mean(np.square(np.clip(d_tilde_metric, 0, 2), np.clip(d_metric, 0, 2))))
    stats_dict['RMSDclamp02(d_star, d)'] = np.sqrt(np.mean(np.square(np.clip(d_star_metric, 0, 2), np.clip(d_metric, 0, 2))))
    stats_dict['RMS(b)'] = np.sqrt(np.mean(np.square(b)))
    stats_dict['RMS(q)'] = np.sqrt(np.mean(np.square(q)))
    return stats_dict


def do_cooccurrence_metric_analysis(dataset_name):
    gts, logits, classnames = load_data(dataset_name)
    probs = 1. / (1. + np.exp(-logits))
    d_metric = compute_metric_from_scores(gts)
    d_hat_metric = compute_metric_from_scores(probs)
    lambda_vec = compute_lambda_vec(gts, probs)
    probs_adjusted = probs - lambda_vec[np.newaxis,:]
    d_tilde_metric = compute_metric_from_scores(probs_adjusted)
    gtmargs = compute_margs_from_scores(gts)
    thresholds = np.array([np.quantile(probs[:,i], 1 - gtmargs[i]) for i in range(probs.shape[1])])
    probs_binarized = (probs > thresholds[np.newaxis,:]).astype('int64')
    d_star_metric = compute_metric_from_scores(probs_binarized)

    #b and q
    mu_vec = compute_mu_vec(gts, probs)
    b = compute_b(gts, mu_vec, lambda_vec)
    r = compute_r(gts, probs, mu_vec, lambda_vec)
    q = compute_q(gts, mu_vec, lambda_vec, r)

    #sanity
    d_hat_metric_sanity = compute_d_hat_metric_sanity(gts, b, q)
    d_tilde_metric_sanity = compute_d_tilde_metric_sanity(gts, q)

    #stats_dict
    stats_dict = compute_stats(d_metric, d_hat_metric, d_tilde_metric, d_star_metric, b, q, d_hat_metric_sanity, d_tilde_metric_sanity)
    print(dataset_name.split('_')[0])
    pprint.pp(stats_dict)

    #write stuff
    analysis_filename = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/%s_cooccurrence_metric_analysis.csv'%(dataset_name.split('_')[0]))
    f = open(analysis_filename, 'w')
    f.write('i,j,classnames[i],classnames[j],d_ij,d_hat_ij,d_tilde_ij,d_star_ij,b_ij,q_ij,lambda_i,mu_i,lambda_j,mu_j,d_hat_ij_sanity,d_tild_ij_sanity\n')
    for i in range(gts.shape[1] - 1):
        for j in range(i + 1, gts.shape[1]):
            f.write('%d,%d,%s,%s,%f,%f,%f,%f,%f,%f,%f,%f\n'%(i, j, classnames[i], classnames[j], d_metric[i,j], d_hat_metric[i,j], d_tilde_metric[i,j], d_star_metric[i,j], b[i,j], q[i,j], lambda_vec[i], mu_vec[i], lambda_vec[j], mu_vec[j], d_hat_metric_sanity[i,j], d_tilde_metric_sanity[i,j]))

    f.close()


def usage():
    print('Usage: python do_cooccurrence_metric_analysis.py <dataset_name>')


if __name__ == '__main__':
    do_cooccurrence_metric_analysis(*(sys.argv[1:]))
