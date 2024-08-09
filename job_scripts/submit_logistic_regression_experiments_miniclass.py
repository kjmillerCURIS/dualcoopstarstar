import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_logistic_regression_experiment.sh'


def submit_logistic_regression_experiments_miniclass():
    for dataset_name in ['COCO2014_partial', 'nuswide_partial']:
        for input_type in ['cossims']:
            for standardize in [0]:
                for balance in [0]:
                    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]:
                        job_name = 'logistic_regression_%s_%s_standardize%d_balance%d_C%s_miniclass1'%(dataset_name.split('_')[0], input_type, standardize, balance, str(C))
                        my_cmd = 'qsub -N %s -v DATASET_NAME=%s,INPUT_TYPE=%s,STANDARDIZE=%d,BALANCE=%d,C=%s,MINICLASS=1 %s'%(job_name, dataset_name, input_type, standardize, balance, str(C), SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_logistic_regression_experiments_miniclass()
