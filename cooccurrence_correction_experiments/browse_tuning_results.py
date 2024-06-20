import os
import sys
import glob
import torch
from tqdm import tqdm


def browse_tuning_results(output_dir):
    results_dict_filenames = sorted(glob.glob(os.path.join(output_dir, '*.pth')))
    for results_dict_filename in results_dict_filenames:
        results_dict = torch.load(results_dict_filename)
        print('')
        print(os.path.basename(results_dict_filename))
        print(results_dict['params'])
        print(results_dict['mAP'])


def usage():
    print('Usage: python browse_tuning_results.py <output_dir>')


if __name__ == '__main__':
    browse_tuning_results(*(sys.argv[1:]))
