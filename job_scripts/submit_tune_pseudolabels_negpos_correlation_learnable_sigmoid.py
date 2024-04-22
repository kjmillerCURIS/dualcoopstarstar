import os
import sys


DEBUG = True
SCRIPT_NAME = 'generic_tune_pseudolabels_negpos_correlation_learnable_sigmoid.sh'


def submit_tune_pseudolabels_negpos_correlation_learnable_sigmoid_runs():
    for alpha in [0.0075, 0.0025, 0.005, 0.0125]: #[0.005, 0.025, 0.0125]: #[0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
        for epsilon in [0.25]:
            for beta_over_alpha in [0.6, 0.9, 1, 1.25]: #[0.25, 0.5, 0.75]:
                beta = alpha * beta_over_alpha
                for zeta in [-0.25]:
                    job_name = 'tune_pseudolabels_negpos_correlation_learnable_sigmoid_alpha%.5f_epsilon%.5f_beta%.5f_zeta%.5f'%(alpha, epsilon, beta, zeta)
                    my_cmd = 'qsub -N %s -v ALPHA=%f,EPSILON=%f,BETA=%f,ZETA=%f %s'%(job_name, alpha, epsilon, beta, zeta, SCRIPT_NAME)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return


if __name__ == '__main__':
    submit_tune_pseudolabels_negpos_correlation_learnable_sigmoid_runs()
