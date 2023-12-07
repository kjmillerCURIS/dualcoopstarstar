import os
import sys
import argparse
from yacs.config import CfgNode as CN


def cfg_example(args):
    cfg = CN()
    cfg.Meow = CN()
    cfg.Meow.MY_INT = 42
    cfg.Meow.MY_STR = 'mrow'
    cfg.merge_from_list(args.opts)
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg_example(args)
