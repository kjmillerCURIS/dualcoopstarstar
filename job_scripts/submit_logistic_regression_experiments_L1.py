import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_logistic_regression_experiment.sh'
OUT_PARENT_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/dualcoopstarstar-data/cooccurrence_correction_experiments/logistic_regression')


def get_output_filename(dataset_name, input_type, standardize, balance, L1, C, miniclass):
    return os.path.join(OUT_PARENT_DIR, 'logistic_regression_%s_%s_standardize%d_balance%d_L1%d_C%s_miniclass%d.pkl'%(dataset_name.split('_')[0], input_type, standardize, balance, L1, str(C), miniclass))


def submit_logistic_regression_experiments_L1():
    for dataset_name in ['COCO2014_partial', 'nuswide_partial']:
        for input_type in ['cossims', 'probs', 'log_probs', 'logits']:
            for standardize in [0, 1]:
                for balance in [0, 1]:
                    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]:
                        output_filename = get_output_filename(dataset_name, input_type, standardize, balance, 1, C, 0)
                        if os.path.exists(output_filename):
                            continue

                        job_name = 'logistic_regression_%s_%s_standardize%d_balance%d_L11_C%s_miniclass0'%(dataset_name.split('_')[0], input_type, standardize, balance, str(C))
                        my_cmd = 'qsub -N %s -v DATASET_NAME=%s,INPUT_TYPE=%s,STANDARDIZE=%d,BALANCE=%d,L1=1,C=%s,MINICLASS=0 %s'%(job_name, dataset_name, input_type, standardize, balance, str(C), SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_logistic_regression_experiments_L1()
