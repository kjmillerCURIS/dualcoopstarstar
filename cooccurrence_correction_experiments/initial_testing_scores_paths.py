import os
import sys


PSEUDOLABEL_TESTING_LOGITS_FILENAME_DICT = {k : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output/TESTINGDUMP_zsclip_ensemble80_%s/Caption_tri_wta_soft_pseudolabel/rn101%s/nctx21_cscTrue_ctpend/seed1/results/testing_logits_dict.pkl'%{'COCO2014_partial' : ('coco', ''), 'nuswide_partial' : ('nuswide', '_nus'), 'VOC2007_partial' : ('voc2007', '')}[k]) for k in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
PSEUDOLABEL_TESTING_COSSIMS_FILENAME_DICT = {k : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/output/TESTINGDUMP_zsclip_ensemble80_%s/Caption_tri_wta_soft_pseudolabel/rn101%s/nctx21_cscTrue_ctpend/seed1/results/testing_cossims_dict.pkl'%{'COCO2014_partial' : ('coco', ''), 'nuswide_partial' : ('nuswide', '_nus'), 'VOC2007_partial' : ('voc2007', '')}[k]) for k in ['COCO2014_partial', 'nuswide_partial', 'VOC2007_partial']}
