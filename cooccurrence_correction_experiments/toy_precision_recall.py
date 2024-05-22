import os
import sys
import numpy as np
from sklearn.metrics import average_precision_score
from compute_mAP import average_precision


def generate_data():
    scores = np.random.uniform(-3, 3, 4800)
    probs = 1 / (1 + np.exp(-scores))
    targets = (np.random.uniform(0, 1, 4800) < probs).astype('int')
    return scores, targets


def toy_precision_recall():
    scores, targets = generate_data()
    my_mAP = average_precision(scores, targets)
    their_mAP = average_precision_score(targets, scores)
    print(my_mAP)
    print(their_mAP)
    print(my_mAP - their_mAP)


if __name__ == '__main__':
    toy_precision_recall()
