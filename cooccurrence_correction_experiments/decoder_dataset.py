import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from harvest_training_gts import TRAINING_GTS_FILENAME_DICT
from compute_initial_pseudolabels import PSEUDOLABEL_LOGITS_FILENAME_DICT
from compute_initial_cossims import PSEUDOLABEL_COSSIMS_FILENAME_DICT
from harvest_testing_gts import TESTING_GTS_FILENAME_DICT
from compute_initial_testing_pseudolabels import PSEUDOLABEL_TESTING_LOGITS_FILENAME_DICT
from compute_initial_testing_cossims import PSEUDOLABEL_TESTING_COSSIMS_FILENAME_DICT


class DecoderDataset(Dataset):

    def __init__(self, dataset_name, train_or_test, params):
        p = params
        assert(train_or_test in ['train', 'test'])
        self.input_type = p['input_type']
        assert(self.input_type in ['cossims', 'probs', 'logits'])
        if train_or_test == 'train':
            with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
                self.gts = pickle.load(f)
        else:
            with open(TESTING_GTS_FILENAME_DICT[dataset_name], 'rb') as f:
                self.gts = pickle.load(f)

        if train_or_test == 'train':
            scores_filename_dict = (PSEUDOLABEL_LOGITS_FILENAME_DICT if self.input_type in ['probs', 'logits'] else PSEUDOLABEL_COSSIMS_FILENAME_DICT)
        else:
            scores_filename_dict = (PSEUDOLABEL_TESTING_LOGITS_FILENAME_DICT if self.input_type in ['probs', 'logits'] else PSEUDOLABEL_TESTING_COSSIMS_FILENAME_DICT)
        
        with open(scores_filename_dict[dataset_name], 'rb') as f:
            self.scores = pickle.load(f)

        if self.input_type == 'probs':
            self.scores = {impath : 1. / (1. + np.exp(-self.scores[impath])) for impath in sorted(self.scores.keys())}

        self.impaths = sorted(self.gts.keys())
        assert(sorted(self.scores.keys()) == self.impaths)

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        impath = self.impaths[idx]
        return {'scores' : torch.tensor(self.scores[impath], dtype=torch.float32), 'gts' : torch.tensor(self.gts[impath], dtype=torch.float32), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def get_impaths(self, idxs):
        return [self.impaths[idx] for idx in idxs]
