import warnings
warnings.filterwarnings("ignore")

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.voc2007_distill
import datasets.coco2014_distill
import datasets.nuswide_distill
import datasets.voc2007_partial
import datasets.coco2014_partial
import datasets.nuswide_trainset_gt
import datasets.nuswide_trainset_gt2

import trainers.zsclip
#import trainers.Caption_distill_double
#import trainers.Caption_dual
#import trainers.Caption_tri
#import trainers.Caption_tri_wta
#import trainers.Caption_tri_wta_soft
#import trainers.Caption_tri_wta_soft_ada
import trainers.dualcoopstarstar_fullpartial


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed > -1:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.DualCoOpStarStar = CN()
    cfg.TRAINER.DualCoOpStarStar.N_CTX = 16  # number of context vectors
    cfg.TRAINER.DualCoOpStarStar.CSC = False  # class-specific context
    cfg.TRAINER.DualCoOpStarStar.CTX_INIT = ""  # initialization words
    cfg.TRAINER.DualCoOpStarStar.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.DualCoOpStarStar.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.DualCoOpStarStar.FIXED_PROMPT_KEY = 'ensemble_80'
    cfg.TRAINER.DualCoOpStarStar.PROMPT_MODE = 'pos_only_fixed_prompt'
    cfg.TRAINER.DualCoOpStarStar.NUM_QUERIES_PER_CLASS = 2
    cfg.TRAINER.DualCoOpStarStar.NUM_ENCODER_LAYERS = 0
    cfg.TRAINER.DualCoOpStarStar.NUM_DECODER_LAYERS = 2
    cfg.TRAINER.DualCoOpStarStar.USE_POSITIONAL_EMBEDDING_IN_LOWER_PART = 1
    cfg.TRAINER.DualCoOpStarStar.USE_POSITIONAL_EMBEDDING_IN_UPPER_PART = 1
    cfg.TRAINER.DualCoOpStarStar.TEMPERATURE_IS_LEARNABLE = 0

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SAMPLE = 0 # Sample some of all datas, 0 for no sampling i.e. using all
    cfg.DATASET.partial_prob = 0.5

    cfg.TRAIN.LOSSFUNC = 'hungarian'
    cfg.TRAIN.LOSS_AVERAGING_MODE = 'mean_all'
    cfg.TRAIN.MATCH_LOSS_FN_TYPE = 'crossent'
    cfg.TRAIN.OPT_LOSS_FN_TYPE = 'ASL'
    cfg.TRAIN.IF_LEARN_SCALE = False
    cfg.TRAIN.IF_LEARN_spatial_SCALE = False
    cfg.TRAIN.spatial_SCALE_text = 50
    cfg.TRAIN.spatial_SCALE_image = 50
    cfg.TRAIN.IF_ablation = False
    cfg.TRAIN.Caption_num = 0

    cfg.TEST.EVALUATOR = 'MLClassificationDualCoOpStarStar'
    cfg.TEST.EVALUATOR_ACT = 'hungarian'
    cfg.TEST.SAVE_PREDS = ""
    
    # several param for spacific transform setting
    cfg.INPUT.random_resized_crop_scale = (0.8, 1.0)
    cfg.INPUT.cutout_proportion = 0.4
    cfg.INPUT.TRANSFORMS_TEST = ("resize", "center_crop", "normalize")

    cfg.TRAIN.CHECKPOINT_FREQ = 1

    #special flags 'n stuff
    cfg.COMPUTE_RANDOM_CHANCE = 0
    cfg.DETR_CHEAT_MODE = 'none'
    cfg.HUNGARIAN_CHEAT_MODE = 'none'


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    cfg.MODE = args.mode
    # 1. From the dataset config file
    if args.dataset_config_file:
        print('merge_from_file {}'.format(args.dataset_config_file))
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    if args.compute_random_chance:
        cfg.COMPUTE_RANDOM_CHANCE = 1

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.compute_random_chance:
        trainer.test()
        return

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--mode", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument('--compute-random-chance', action='store_true', help='give (uniform) random logits, see how they perform')
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
