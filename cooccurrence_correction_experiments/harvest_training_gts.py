import os
import sys
import numpy as np
import pickle
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN
sys.path.append('.')
sys.path.append('Dassl.pytorch-master')
from dassl.config import get_cfg_default
from dassl.data import DataManager
import datasets.coco2014_partial
import datasets.nuswide_trainset_gt
import datasets.nuswideTheirVersion
import datasets.voc2007_partial
sys.path.pop()
sys.path.pop()


NUM_CLASSES_DICT = {'COCO2014_partial' : 80, 'nuswide_partial' : 81, 'VOC2007_partial' : 20, 'nuswideTheirVersion_partial' : 81}
DATASET_CONFIG_FILE_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/datasets/coco2014_partial.yaml'), 'nuswide_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/datasets/nuswide_partial.yaml'), 'VOC2007_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/datasets/voc2007_partial.yaml'), 'nuswideTheirVersion_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/datasets/nuswideTheirVersion_partial.yaml')}
CONFIG_FILE_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_tri_wta_soft_pseudolabel/rn101.yaml'), 'nuswide_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_tri_wta_soft_pseudolabel/rn101_nus.yaml'), 'VOC2007_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_tri_wta_soft_pseudolabel/rn101_bn96.yaml'), 'nuswideTheirVersion_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_tri_wta_soft_pseudolabel/rn101_nus.yaml')}
CONFIG_FILE_DICT_TAIDPT = {'COCO2014_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_distill_double/rn50_coco2014.yaml'), 'VOC2007_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_distill_double/rn50_voc2007.yaml'), 'nuswideTheirVersion_partial' : os.path.expanduser('~/data/dualcoopstarstar/configs/trainers/Caption_distill_double/rn50_nuswide.yaml')}
MOD_DICT = {'COCO2014_partial' : 'pos_norm', 'nuswide_partial' : 'pos200', 'VOC2007_partial' : 'pos200', 'nuswideTheirVersion_partial' : 'pos200'}
DATA_ROOT = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data')
TRAINING_GTS_FILENAME_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_training_gts.pkl'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/nuswide_training_gts.pkl'), 'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/voc2007_training_gts.pkl')}
TESTING_GTS_FILENAME_DICT = {'COCO2014_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/mscoco_testing_gts.pkl'), 'nuswide_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/nuswide_testing_gts.pkl'), 'VOC2007_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/voc2007_testing_gts.pkl'), 'nuswideTheirVersion_partial' : os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/nuswideTheirVersion_testing_gts.pkl')}


def get_cfg(dataset_name, partial_prob=1.0, random_seed=1, model_type=None, is_taidpt=False):

    #setup_cfg():
    cfg = get_cfg_default()

    #extend_cfg():
    cfg.TRAINER.Caption = CN()
    cfg.TRAINER.Caption.N_CTX = 16  # number of context vectors
    cfg.TRAINER.Caption.CSC = False  # class-specific context
    cfg.TRAINER.Caption.CTX_INIT = ""  # initialization words
    cfg.TRAINER.Caption.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.Caption.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.Caption.GL_merge_rate = 0.5
    cfg.TRAINER.Caption.USE_BIAS = 0
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SAMPLE = 0 # Sample some of all datas, 0 for no sampling i.e. using all
    cfg.DATASET.partial_prob = partial_prob
    cfg.TRAIN.IF_LEARN_SCALE = False
    cfg.TRAIN.IF_LEARN_spatial_SCALE = False
    cfg.TRAIN.spatial_SCALE_text = 50
    cfg.TRAIN.spatial_SCALE_image = 50
    cfg.TRAIN.IF_ablation = False
    cfg.TRAIN.Caption_num = 0
    cfg.TRAIN.PSEUDOLABEL_INIT_METHOD = 'global_only'
    cfg.TRAIN.PSEUDOLABEL_INIT_PROMPT_KEY = 'ensemble_80'
    cfg.TRAIN.DO_ADJUST_LOGITS = 0
    cfg.TRAIN.ADJUST_LOGITS_MIN_BIAS = -2.0
    cfg.TRAIN.ADJUST_LOGITS_MAX_BIAS = 5.0
    cfg.TRAIN.ADJUST_LOGITS_TARGET = 2.89
    cfg.TRAIN.ADJUST_LOGITS_EPSILON = 0.05
    cfg.TRAIN.ADJUST_LOGITS_MAXITER = 10
    cfg.TRAIN.PSEUDOLABEL_UPDATE_MODE = 'gaussian_grad'
    cfg.TRAIN.PSEUDOLABEL_UPDATE_GAUSSIAN_BANDWIDTH = 0.2 #options will be 0.1, 0.2, 0.4
    cfg.TRAIN.PSEUDOLABEL_UPDATE_STEPSIZE = 0.25 #options will be 0.0625, 0.125, 0.25, 0.5, 1.0
    cfg.TRAIN.PSEUDOLABEL_UPDATE_FREQ = 1
    cfg.TRAIN.LOSSFUNC = 'crossent'
    cfg.TEST.EVALUATOR = 'MLClassificationDualCoOpStarStar'
    cfg.TEST.EVALUATOR_ACT = 'default'
    cfg.TEST.SAVE_PREDS = ""
    cfg.INPUT.random_resized_crop_scale = (0.8, 1.0)
    cfg.INPUT.cutout_proportion = 0.4
    cfg.INPUT.TRANSFORMS_TEST = ("resize", "center_crop", "normalize")
    cfg.TRAIN.CHECKPOINT_FREQ = 1
    cfg.COMPUTE_RANDOM_CHANCE = 0
    cfg.COMPUTE_ZSCLIP = 0
    cfg.ZSCLIP_USE_COSSIM = 0
    cfg.EVAL_TRAINING_PSEUDOLABELS = 0

    #back to setup_cfg():
    cfg.MODE = MOD_DICT[dataset_name]

    #merge_from_file() dataset:
    cfg.merge_from_file(DATASET_CONFIG_FILE_DICT[dataset_name])

    #merge_from_file() config:
    if is_taidpt:
        cfg.merge_from_file(CONFIG_FILE_DICT_TAIDPT[dataset_name])
    else:
        cfg.merge_from_file(CONFIG_FILE_DICT[dataset_name])

    #reset_cfg():
    cfg.DATASET.ROOT = DATA_ROOT
    cfg.SEED = random_seed
    cfg.TRAINER.NAME = 'Caption_tri_wta_soft_pseudolabel'

    #merge_from_list():
    cfg.TRAINER.Caption.N_CTX = 21
    cfg.TRAINER.Caption.CSC = True
    cfg.TRAINER.Caption.CLASS_TOKEN_POSITION = 'end'
    cfg.TRAINER.Caption.USE_BIAS = 1
    cfg.TRAIN.DO_ADJUST_LOGITS = 0
    cfg.TRAIN.PSEUDOLABEL_UPDATE_GAUSSIAN_BANDWIDTH = 0.2
    cfg.TRAIN.PSEUDOLABEL_UPDATE_STEPSIZE = 0.0

    if model_type is not None:
        cfg.MODEL.BACKBONE.NAME = model_type.replace('ViT-B', 'ViT-B/').replace('ViT-L', 'ViT-L/').replace('14336px', '14@336px')
        if model_type == 'ViT-L14336px':
            cfg.INPUT.SIZE = (336, 336)
        elif 'ViT-B' in model_type or 'ViT-L' in model_type:
            cfg.INPUT.SIZE = (224, 224)

    cfg.freeze()

    return cfg


def get_data_manager(dataset_name, cfg=None, is_taidpt=False):
    if cfg is None:
        cfg = get_cfg(dataset_name, is_taidpt=is_taidpt)

    dm = DataManager(cfg, skip_train=(dataset_name=='nuswideTheirVersion_partial'))
    return dm


#d[impath] = labels
#labels will be in {0, 1}^N
def harvest_training_gts(dataset_name):
    assert(dataset_name in TRAINING_GTS_FILENAME_DICT)
    dm = get_data_manager(dataset_name)
    labels = {}
    for item in tqdm(dm.dataset.train_x):
        assert(item.impath not in labels)
        assert(item.label.shape == (NUM_CLASSES_DICT[dataset_name],))
        assert(np.all((item.label == 1) | (item.label == -1)))
        labels[item.impath] = np.maximum(item.label, 0) #-1 ==> 0, +1 ==> 1

    with open(TRAINING_GTS_FILENAME_DICT[dataset_name], 'wb') as f:
        pickle.dump(labels, f)


#d[impath] = labels
#labels will be in {0, 1}^N
def harvest_testing_gts(dataset_name):
    assert(dataset_name in TESTING_GTS_FILENAME_DICT)
    print(TESTING_GTS_FILENAME_DICT[dataset_name])
    dm = get_data_manager(dataset_name)
    labels = {}
    for item in tqdm(dm.dataset.test):
        assert(item.impath not in labels)
        label = item.label
        assert(label.shape == (NUM_CLASSES_DICT[dataset_name],))
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        if dataset_name == 'nuswideTheirVersion_partial':
            assert(np.all((label == 1) | (label == 0)))
        else:
            assert(np.all((label == 1) | (label == -1)))

        labels[item.impath] = np.maximum(label, 0) #-1 ==> 0, +1 ==> 1

    with open(TESTING_GTS_FILENAME_DICT[dataset_name], 'wb') as f:
        pickle.dump(labels, f)


def usage():
    print('Usage: python harvest_training_gts.py <dataset_name>')


if __name__ == '__main__':
    #harvest_training_gts(*(sys.argv[1:]))
    harvest_testing_gts(*(sys.argv[1:]))
