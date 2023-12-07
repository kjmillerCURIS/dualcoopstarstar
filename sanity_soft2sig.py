import os
import sys
import numpy as np
import torch


def softlogits2siglogits(softlogits):
    #how to turn softmax logits into sigmoid logits?
    #probs[i] = exp(-softlogits[i]) / sum_j exp(-softlogits[j])
    #probs[i] = 1 / (1 + exp(-siglogits[i]))
    #...
    #siglogits[i] = log(exp(softlogits[i]) / (sum_j exp(softlogits[j]) - exp(softlogits[i])))
    #             = softlogits[i] - log(sum_j exp(softlogits[j]) - exp(softlogits[i]))
    #best way to make it numerically stable is probably just to subtract the max of softlogits
    softlogits = softlogits - torch.max(softlogits, 1, keepdim=True)[0]
    sumexps = torch.sum(torch.exp(softlogits), 1, keepdim=True)
    siglogits = softlogits - torch.log(sumexps - torch.exp(softlogits))
    return siglogits


def sanity_soft2sig():
    softlogits = torch.tensor(np.random.randn(64, 80).astype('float32'))
    siglogits = softlogits2siglogits(softlogits)
    softprobs = torch.softmax(softlogits, dim=1)
    sigprobs = torch.sigmoid(siglogits)
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    sanity_soft2sig()
