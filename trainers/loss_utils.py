from numpy import deprecate
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast


POS_CROSSENT_LOSS_FN = lambda logits : F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
NEG_CROSSENT_LOSS_FN = lambda logits : F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
POS_NEGPROB_LOSS_FN = lambda logits : -torch.sigmoid(logits)
NEG_NEGPROB_LOSS_FN = lambda logits : -(1 - torch.sigmoid(logits))


''' SOME USEFUL LOGIT MATH: '''
''' if you happen to have 2 logits, i.e. logits_pos and logits_neg, then you can combine them into one as follows: '''
''' p = exp(logits_pos) / (exp(logits_neg) + exp(logits_pos)) '''
''' p = 1 / (exp(logits_neg - logits_pos) + 1) '''
''' p = 1 / (1 + exp(-(logits_pos - logits_neg))) '''
''' Therefore, logits = logits_pos - logits_neg '''


''' can apply this to a logits tensor, which we expect to have 3 dimensions (but ideally we could handle any number of dimensions) '''
''' they're "logits" in the sense that doing nn.Sigmoid()(logits) should get you the probabilities '''
class AsymmetricLoss_softmax_partial_fn(nn.Module):

    #just keep all the hyperparams same as DualCoOp++ for now
    #gt should be +1 or -1, depending on whether we want to encourage high
    def __init__(self, gt, gamma_neg=2, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss_softmax_partial_fn, self).__init__()
        assert(gt > 0 or gt < 0)
        self.gt = gt
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits):
        
        #compute probs, which will be high if logits lines up with self.gt (i.e. we subtract it from 1 if self.gt < 0)
        probs = self.sigmoid(logits)
        if self.gt < 0:
            probs = (probs - self.clip).clamp(min=0)
            probs = 1 - probs
        else:
            assert(self.gt > 0)

        #compute log part - note that we can just use probs regardless of self.gt
        log_part = torch.log(probs.clamp(min=self.eps))

        #compute multiplier part
        if self.gt > 0:
            gamma = self.gamma_pos
        else:
            assert(self.gt < 0)
            gamma = self.gamma_neg

        if gamma <= 0:
            assert(gamma == 0)
            return -log_part  #just normal loss, with maybe a negative probability shift

        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                multiplier_part = torch.pow(1 - probs, gamma) #no matter what self.gt is, 1 - probs will be the opposite of what we fed to the log part
        
        else:
            multiplier_part = torch.pow(1 - probs, gamma)

        return -multiplier_part * log_part


#return pos_match_loss_fn, neg_match_loss_fn
#match_loss_fn_type can be "crossent" (exactly what it sounds like) or "neg_prob" (-p or -(1-p))
def get_match_loss_fns(match_loss_fn_type):
    assert(match_loss_fn_type in ['crossent', 'neg_prob'])
    if match_loss_fn_type == 'crossent':
        return POS_CROSSENT_LOSS_FN, NEG_CROSSENT_LOSS_FN
    elif match_loss_fn_type == 'neg_prob':
        return POS_NEGPROB_LOSS_FN, NEG_NEGPROB_LOSS_FN
    else:
        assert(False)



#return pos_opt_loss_fn, neg_opt_loss_fn
#opt_loss_fn_type can be "crossent" (exactly what it sounds like) or "ASL" (the thingy from DualCoOp++)
def get_opt_loss_fns(opt_loss_fn_type):
    assert(opt_loss_fn_type in ['crossent', 'ASL'])
    if opt_loss_fn_type == 'crossent':
        return POS_CROSSENT_LOSS_FN, NEG_CROSSENT_LOSS_FN
    elif opt_loss_fn_type == 'ASL':
        return AsymmetricLoss_softmax_partial_fn(1), AsymmetricLoss_softmax_partial_fn(-1)
    else:
        assert(False)
