import os
import sys
from harvest_training_gts import get_cfg
from compute_cossims_test_noaug import get_clip_model


def mrow():
    cfg = get_cfg('VOC2007_partial', model_type='RN50')
    clip_model = get_clip_model(cfg)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    mrow()
