import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DecoderModel(nn.Module):

    #you'll need to compute or load standardization_info externally
    def __init__(self, num_classes, standardization_info, params):
        super().__init__()
        p = params
        layers = []
        cur_size = num_classes
        for _ in range(p['num_hidden_layers']):
            if p['use_intermediate_dropout']:
                layers.append(nn.Dropout(p=p['intermediate_dropout_prob']))
            
            layers.append(nn.Linear(cur_size, p['hidden_layer_size']))
            cur_size = p['hidden_layer_size']
            if p['use_intermediate_batchnorm']:
                layers.append(nn.BatchNorm1d(cur_size))

            layers.append(nn.ReLU())
            
        if p['use_final_dropout']:
            layers.append(nn.Dropout(p=p['final_dropout_prob']))
            
        layers.append(nn.Linear(cur_size, num_classes))
        cur_size = num_classes
        if p['use_final_batchnorm']:
            layers.append(nn.BatchNorm1d(cur_size))

        self.base = nn.Sequential(*layers)
        self.standardize_input = p['standardize_input']
        if self.standardize_input:
            self.standardization_info = {'means' : standardization_info['means'].clone().detach().cuda(), 'sds' : standardization_info['sds'].clone().detach().cuda()}
            assert(self.standardization_info['means'].shape == (num_classes,))
            assert(self.standardization_info['sds'].shape == (num_classes,))

    #X should be shape (batch_size, num_classes)
    #will return logits of same shape
    def forward(self, X):
        if self.standardize_input:
            X = (X - torch.unsqueeze(self.standardization_info['means'], 0)) / torch.unsqueeze(self.standardization_info['sds'], 0)

        return self.base(X)
