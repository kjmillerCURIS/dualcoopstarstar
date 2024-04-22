import os
import sys
import numpy as np


def make_label_getter(my_dataloader):
    for batch in my_dataloader:
        yield batch['label']


#label_getter should be something I can stick into a for-loop and get batches
#should give Pr(X_j=1, X_i=1) and Pr(X_j=1, X_i=0), Pr(X_i)
def make_joint_tables(label_getter, classnames):
    num_classes = len(classnames)
    joint_table_pos = np.zeros(num_classes, num_classes)
    joint_table_neg = np.zeros(num_classes, num_classes)
    marginal_table = np.zeros(num_classes)
    total = 0
    for label_batch in label_getter:
        assert(len(label_batch.shape) == 2)
        assert(label_batch.shape[1] == num_classes)
        joint_table_pos += label_batch.T @ label_batch
        joint_table_neg += (1 - label_batch.T) @ label_batch
        marginal_table += np.sum(label_batch, axis=0)
        total += label_batch.shape[0]

    joint_table_pos /= total
    joint_table_neg /= total
    marginal_table /= total

    return joint_table_pos, joint_table_neg, marginal_table


def make_cond_tables(joint_table_pos, joint_table_neg, marginal_table):
    cond_table_pos = joint_table_pos / marginal_table[np.newaxis,:]
    cond_table_neg = joint_table_neg / (1 - marginal_table[np.newaxis,:])
    return cond_table_pos, cond_table_neg


def plot_table(my_table_pos, my_table_neg, classnames, my_title, plot_filename):
    assert(False)


def plot_marginal_table(marginal_table, classnames, plot_filename):
    assert(False)
