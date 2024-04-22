import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_nuswide_frozen_pseudolabel_with_skipping.sh'


def submit_frozen_pseudolabel_runs_with_skipping_nuswide():
    job_name = 'frozen_pseudolabel_with_skipping_nuswide_seed1'
    my_cmd = 'qsub -N %s -v run_ID=%s %s'%(job_name, job_name, SCRIPT_NAME)
    print('submitting training run: "%s"'%(my_cmd))
    os.system(my_cmd)
    if DEBUG:
        print('DEBUG MODE: let\'s see how that first run goes...')
        return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_with_skipping_nuswide()
