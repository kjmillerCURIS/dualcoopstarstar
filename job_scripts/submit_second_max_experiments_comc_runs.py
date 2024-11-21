import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_second_max_experiments_CoMC.sh'
DATASET_NAME_LIST = ['COCO2014_partial']
SEED_LIST = [1]


def submit_second_max_experiments_comc_runs():
    for seed in SEED_LIST:
        for dataset_name in DATASET_NAME_LIST:
            for is_strong in [0,1]:
                for use_rowcalibbase in [0,1]:
                    job_name = 'secondmax_CoMC_%s_rowcalibbase%d_strong%d_seed%d'%(dataset_name.split('_')[0], use_rowcalibbase, is_strong, seed)
                    my_cmd = 'qsub -N %s -v DATASET_NAME=%s,USE_ROWCALIBBASE=%d,US_STRONG=%d,THEM_STRONG=%d,SEED=%d %s'%(job_name, dataset_name, use_rowcalibbase, is_strong, is_strong, seed, SCRIPT_NAME)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return


if __name__ == '__main__':
    submit_second_max_experiments_comc_runs()
