import os
import sys
import tiktoken
from harvest_mscoco_training_gts import get_data_manager


if __name__ == '__main__':
    dm = get_data_manager()
    classnames = dm.dataset.classnames
    s = ', '.join(classnames)
    enc = tiktoken.get_encoding("cl100k_base")
    meow = enc.encode(s)
    import pdb
    pdb.set_trace()
