import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_tune_pseudolabels_negpospenalty.sh'


def submit_tune_pseudolabels_negpospenalty_runs():
    for alpha in [0.3, 0.4, 0.6]:
        for epsilon in [0.25]:
            for beta in [0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
                for zeta in [-0.25]: #[-0.75, -0.5, -0.25]:
                    for negpos_cost_type in ['prob_stopgrad_logit']:
                        job_name = 'tune_pseudolabels_negpenalty_alpha%.5f_epsilon%.5f_beta%.5f_zeta%.5f_cost_%s'%(alpha, epsilon, beta, zeta, negpos_cost_type)
                        my_cmd = 'qsub -N %s -v ALPHA=%f,EPSILON=%f,BETA=%f,ZETA=%f,NEGPOS_COST_TYPE=%s %s'%(job_name, alpha, epsilon, beta, zeta, negpos_cost_type, SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_tune_pseudolabels_negpospenalty_runs()
