import os
import sys
import numpy as np
import torch
from loss_utils import get_opt_loss_fns
from utils import dualcoop_softmax_loss


#inputs should be (BATCH_SIZE, 2, NUM_CLASSES)
#middle dimension should be in (neg, pos) order
#def dualcoop_softmax_loss(inputs, inputs_g, targets):


BATCH_SIZE = 64
NUM_CLASSES = 80
NOISE_SD = 0.01
RANDOM_SEED = 0


#returns logits_pos, logits_neg, annotations
def make_data():
    #np.random.seed(RANDOM_SEED)
    annotations = np.ones((BATCH_SIZE, NUM_CLASSES))
    annotations[:,NUM_CLASSES//2:] = -1
    logits_pos = annotations + NOISE_SD * np.random.randn(BATCH_SIZE, NUM_CLASSES)
    logits_neg = -annotations + NOISE_SD * np.random.randn(BATCH_SIZE, NUM_CLASSES)
    annotations[:,NUM_CLASSES//4:-NUM_CLASSES//4] = 0 #partial
    return torch.tensor(logits_pos), torch.tensor(logits_neg), torch.tensor(annotations)


def loss_sanity_check():
    logits_pos, logits_neg, annotations = make_data()
    pos_loss_fn, neg_loss_fn = get_opt_loss_fns('ASL')

    logits = logits_pos - logits_neg
    pos_losses = pos_loss_fn(logits)
    neg_losses = neg_loss_fn(logits)
    print('my pos_losses:')
    print(pos_losses.cpu().numpy()[0])
    print('\nmy neg_losses:')
    print(neg_losses.cpu().numpy()[0])
    my_loss = torch.mean(pos_losses * (annotations > 0).to(torch.int64) + neg_losses * (annotations < 0).to(torch.int64))

    inputs = torch.cat([logits_neg[:,None,:], logits_pos[:,None,:]], dim=1)
    print('\ntheir losses:')
    their_loss = dualcoop_softmax_loss(inputs, None, annotations, verbose=True)

    print('\nmy_loss: ' + str(my_loss.item()))
    print('my_loss (their averaging): ' + str(my_loss.item() * BATCH_SIZE * NUM_CLASSES * 0.02))
    print('their_loss: ' + str(their_loss.item()))


if __name__ == '__main__':
    loss_sanity_check()
