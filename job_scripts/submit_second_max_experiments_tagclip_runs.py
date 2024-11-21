import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_second_max_experiments_TagCLIP.sh'
DATASET_NAME_LIST = ['VOC2007_partial', 'COCO2014_partial']


def submit_second_max_experiments_tagclip_runs():
    for dataset_name in DATASET_NAME_LIST:
        for use_log in [0,1]:
            for use_rowcalibbase in [0,1]:
                job_name = 'secondmax_TagCLIP_%s_log%d_rowcalibbase%d'%(dataset_name.split('_')[0], use_log, use_rowcalibbase)
                my_cmd = 'qsub -N %s -v DATASET_NAME=%s,USE_LOG=%d,USE_ROWCALIBBASE=%d %s'%(job_name, dataset_name, use_log, use_rowcalibbase, SCRIPT_NAME)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return


if __name__ == '__main__':
    submit_second_max_experiments_tagclip_runs()
