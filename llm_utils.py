import os
import sys
import pickle
from generate_classname2list import get_out_filename_helper
#get_out_filename_helper(llm_name, dataset_name, create_dirs=True)


def get_classname_lists(classnames, cfg):
    dataset_name = cfg.DATASET.NAME
    llm_name = cfg.llm_name
    out_filename = get_out_filename_helper(llm_name, dataset_name, create_dirs=False)
    with open(out_filename, 'rb') as f:
        classname2list = pickle.load(f)

    return [classname2list[classname] for classname in classnames]
