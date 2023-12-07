import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_utils import POS_CROSSENT_LOSS_FN, NEG_CROSSENT_LOSS_FN



class HungarianMatcher(nn.Module):

    #pos_<match/opt>_loss_fn and neg_<match/opt>_loss_fn should take in logits and return losses (assuming you plug logits into sigmoid to get probabilities)
    #pos_<match/opt>_loss_fn should encourage high probabilities, neg_<match/opt>_loss_fn should encourage low probabilities
    #the "match" versions will be used by self.match_phase(), while the "opt" versions will be used by self.optimization_phase()
    #the difference is that the "opt" versions might do stuff to e.g. downweight easy negatives and upweight hard ones
    #you don't necessarily want the matching phase to be influenced by that, better to have it just maximize the log-likelihood (or sum-likelihood if you so choose)
    def __init__(self, pos_match_loss_fn, neg_match_loss_fn, pos_opt_loss_fn, neg_opt_loss_fn, hungarian_cheat_mode='none', loss_averaging_mode='mean_all'):
        super().__init__()
        self.pos_match_loss_fn = pos_match_loss_fn
        self.neg_match_loss_fn = neg_match_loss_fn
        self.pos_opt_loss_fn = pos_opt_loss_fn
        self.neg_opt_loss_fn = neg_opt_loss_fn
        self.hungarian_cheat_mode = hungarian_cheat_mode
        self.loss_averaging_mode = loss_averaging_mode
        assert(self.hungarian_cheat_mode in ['none','fixed_match_full_loss','fixed_match_diagonal_loss'])
        assert(self.loss_averaging_mode in ['mean_all','mean_except_class'])

    #returns cost_matrix_cat, pos_class_indices_cat, sizes
    #where cost_matrix_cat has shape (num_queries, total_num_pos_classes) and is the cost matrices concatenated over the last dimension
    #and pos_class_indices_cat maps column indices from the chunks of cost_matrix_cat to indices in annotations
    #sizes is a list of sizes telling us how to split cost_matrix_cat and pos_class_indices_cat
    #these will all be numpy arrays on the CPU
    def _form_cost_matrix(self, pos_losses, neg_losses, annotations):
        cost_matrix_list = []
        pos_indices_list = []
        sizes = []
        for pos_losses_one, neg_losses_one, annotations_one in zip(pos_losses, neg_losses, annotations):
            if not torch.any(annotations_one > 0).item():
                sizes.append(0)
                continue

            pos_match_part = (pos_losses_one - neg_losses_one)[:, (annotations_one > 0)]
            query_taken_part = torch.sum(neg_losses_one[:,(annotations_one == 0)], -1, keepdim=True)
            assert(query_taken_part.shape == (pos_losses.shape[1], 1))
            cost_matrix = pos_match_part + query_taken_part
            pos_indices = (annotations_one > 0).nonzero().squeeze(1)
            cost_matrix_list.append(cost_matrix)
            pos_indices_list.append(pos_indices)
            sizes.append(len(pos_indices))

        cost_matrix_cat = torch.cat(cost_matrix_list, -1)
        pos_indices_cat = torch.cat(pos_indices_list, -1)
        return cost_matrix_cat, pos_indices_cat, sizes

    #logits should be shape (batch_size, num_queries, num_classes). It will be plugged into binary sigmoid.
    #annotations should be shape (batch_size, num_classes) and be +1, -1, 0 notation
    #will return matches, which is shape (batch_size, num_queries, num_classes) and is a binary mask
    #matches will be a tensor on the GPU
    def match_phase(self, logits, annotations):
        assert(len(logits.shape) == 3)
        assert(annotations.shape == (logits.shape[0], logits.shape[2]))
        if self.hungarian_cheat_mode in ['fixed_match_full_loss', 'fixed_match_diagonal_loss']:
            assert(logits.shape[1] == logits.shape[2])
            with torch.no_grad():
                matches_one_list = []
                for annotations_one in annotations:
                    matches_one = torch.diag((annotations_one > 0).to(torch.int64))
                    matches_one_list.append(matches_one)

                return torch.stack(matches_one_list, dim=0)
        else:
            assert(self.hungarian_cheat_mode == 'none')

        with torch.no_grad():
            pos_losses = self.pos_match_loss_fn(logits)
            neg_losses = self.neg_match_loss_fn(logits)
            cost_matrix_cat, pos_class_indices_cat, sizes = self._form_cost_matrix(pos_losses, neg_losses, annotations)
            assert(len(sizes) == logits.shape[0])
            matches_list = []
            for cost_matrix, pos_class_indices in zip(cost_matrix_cat.cpu().split(sizes, -1), pos_class_indices_cat.cpu().split(sizes, -1)):
                assert(len(cost_matrix.shape) == 2)
                assert(len(pos_class_indices.shape) == 1)
                assert(cost_matrix.shape[0] == logits.shape[1])
                my_matches = np.zeros(logits.shape[1:])
                if cost_matrix.shape[-1] == 0:
                    matches_list.append(my_matches)
                    continue

                i_list, j_list = linear_sum_assignment(cost_matrix)
                j_list = pos_class_indices[j_list]
                my_matches[i_list, j_list] = 1
                matches_list.append(my_matches)

            return torch.tensor(np.array(matches_list)).to(logits.device)

    #logits should be shape (batch_size, num_queries, num_classes). It will be plugged into binary sigmoid.
    #annotations should be shape (batch_size, num_classes) and be +1, -1, 0 notation
    #matches should be shape (batch_size, num_queries, num_classes) and is a binary mask
    #will return loss, which is differentiable scalar
    def optimization_phase(self, logits, annotations, matches):
        assert(len(logits.shape) == 3)
        assert(logits.shape == matches.shape)
        assert(annotations.shape == (logits.shape[0], logits.shape[2]))

        pos_losses = self.pos_opt_loss_fn(logits)
        neg_losses = self.neg_opt_loss_fn(logits)

        if self.hungarian_cheat_mode == 'fixed_match_diagonal_loss':
            assert(logits.shape[1] == logits.shape[2])
            pos_diags = torch.stack([torch.diag(pos_losses_one) for pos_losses_one in pos_losses])
            neg_diags = torch.stack([torch.diag(neg_losses_one) for neg_losses_one in neg_losses])
            if self.loss_averaging_mode == 'mean_all':
                loss = torch.mean(pos_diags * (annotations > 0).to(torch.int64) + neg_diags * (annotations < 0).to(torch.int64))
            else:
                assert(self.loss_averaging_mode == 'mean_except_class')
                loss_terms = pos_diags * (annotations > 0).to(torch.int64) + neg_diags * (annotations < 0).to(torch.int64)
                assert(loss_terms.shape == annotations.shape)
                loss = torch.mean(torch.sum(loss_terms, dim=-1))
            
            return loss
        else:
            assert(self.hungarian_cheat_mode in ['none', 'fixed_match_full_loss'])

        #should have neg-loss on all non-match pairs, EXCEPT (unmatched_query, unknown_class) pairs which should have zero loss
        neg_mask = 1 - matches #start with all non-match pairs
        neg_mask[(annotations[:,None,:] == 0) & torch.ones((logits.shape[0],logits.shape[1],1), dtype=torch.bool, device=logits.device)] = 0 #zero out (*, unknown_class) pairs
        neg_mask[(torch.max(matches, -1, keepdim=True)[0] > 0) & (annotations[:,None,:] == 0)] = 1 #bring back (matched_query, unknown_class) pairs

        #FIXME: figure out if mean over ALL dimensions is actually the right thing to do
        if self.loss_averaging_mode == 'mean_all':
            loss = torch.mean(matches * pos_losses + neg_mask * neg_losses)
        else:
            assert(self.loss_averaging_mode == 'mean_except_class')
            loss = torch.mean(torch.sum(matches * pos_losses + neg_mask * neg_losses, dim=-1))

        return loss

    #calls match_phase() and then optimization_phase() and returns loss and matches
    #see documentations of those functions
    def forward(self, logits, annotations):
        matches = self.match_phase(logits, annotations)
        loss = self.optimization_phase(logits, annotations, matches)
        return loss, matches

    def do_inference(self, logits, allow_more_classes_than_queries=False):
        annotations = torch.ones_like(logits[:,0,:]) #(batch_size, num_classes)
        matches = self.match_phase(logits, annotations) #(batch_size, num_queries, num_classes)
        logits_hungarian = torch.sum(matches * logits, dim=1, keepdim=False) #(batch_size, num_classes)
        if logits.shape[1] < logits.shape[2]:
            assert(allow_more_classes_than_queries)
            make_negative = (torch.sum(matches, dim=1, keepdim=False) == 0)
            logits_hungarian[make_negative] = float('-inf') #let's hope this doesn't blow up the mAP computation...

        return logits_hungarian, matches


def test_hungarian_matcher():

    #toy inputs
    logits = np.random.uniform(-3, 3, (1, 18, 9)).astype('float32')
    annotations = np.array([[1,1,1,0,0,0,-1,-1,-1]]).astype('int32')
    
    #get outputs from method
    hm = HungarianMatcher(POS_CROSSENT_LOSS_FN, NEG_CROSSENT_LOSS_FN, POS_CROSSENT_LOSS_FN, NEG_CROSSENT_LOSS_FN)
    matches = hm.match_phase(torch.tensor(logits), torch.tensor(annotations))
    loss = hm.optimization_phase(torch.tensor(logits), torch.tensor(annotations), matches)
    matches = np.squeeze(matches.numpy())
    loss = np.squeeze(loss.numpy())
    
    #get expected outputs
    best_matches = None
    best_loss = float('+inf')
    for mc0 in range(18):
        for mc1 in range(18):
            for mc2 in range(18):
                if mc0 == mc1 or mc0 == mc2 or mc1 == mc2:
                    continue

                proposed_matches = np.zeros((1, 18, 9)).astype('int32')
                proposed_matches[0, mc0, 0] = 1
                proposed_matches[0, mc1, 1] = 1
                proposed_matches[0, mc2, 2] = 1
                pos_losses = -np.log(1/(1 + np.exp(-logits)))
                neg_losses = -np.log(1-1/(1 + np.exp(-logits)))
                neg_mask = 1 - proposed_matches
                neg_mask[0, :, 3:6] = 0
                neg_mask[0, mc0, 3:6] = 1
                neg_mask[0, mc1, 3:6] = 1
                neg_mask[0, mc2, 3:6] = 1
                proposed_loss = np.mean(proposed_matches * pos_losses) + np.mean(neg_mask * neg_losses)
                if proposed_loss < best_loss:
                    best_loss = proposed_loss
                    best_matches = proposed_matches

    print(loss)
    print(best_loss)
    assert(np.fabs(loss - best_loss) < 1e-6)
    assert(np.all(matches == best_matches))


if __name__ == '__main__':
    test_hungarian_matcher()
