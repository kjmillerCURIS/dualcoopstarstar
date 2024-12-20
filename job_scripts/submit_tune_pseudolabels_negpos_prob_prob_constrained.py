import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_tune_pseudolabels_negpos_prob_prob_constrained.sh'


def submit_tune_pseudolabels_negpos_prob_prob_constrained_runs():
    for alpha in [0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
        for epsilon in [0.25]:
            for beta_over_alpha in [0.25, 0.5, 0.75]:
                beta = alpha * beta_over_alpha
                for zeta in [-0.25]:
                    for gamma in [0.05, 0.1, 0.2]:
                        job_name = 'tune_pseudolabels_negpos_prob_prob_constrained_alpha%.5f_epsilon%.5f_beta%.5f_zeta%.5f_gamma%.5f'%(alpha, epsilon, beta, zeta, gamma)
                        my_cmd = 'qsub -N %s -v ALPHA=%f,EPSILON=%f,BETA=%f,ZETA=%f,GAMMA=%f %s'%(job_name, alpha, epsilon, beta, zeta, gamma, SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_tune_pseudolabels_negpos_prob_prob_constrained_runs()
