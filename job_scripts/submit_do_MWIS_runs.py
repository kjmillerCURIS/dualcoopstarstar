import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_do_MWIS.sh'


def submit_do_MWIS_runs():
    for dataset_name in ['COCO2014_partial']:
        for score_type in ['rank_with_gtmargs', 'rank_without_gtmargs']: #['binary_top1perc']: #['binary_top5perc']: #['binary_top2perc']: #['prob', 'neglogcompprob', 'adaptivelogprob_onemin', 'adaptivelogprob_minperclass']:
            for conflict_threshold in [0.001, 0.01, 0.025, 0.05, 0.1, 0.2]:
                job_name = 'MWIS-%s-%f-%s'%(dataset_name, conflict_threshold, score_type)
                my_cmd = 'qsub -N %s -v DATASET_NAME=%s,CONFLICT_THRESHOLD=%f,SCORE_TYPE=%s %s'%(job_name, dataset_name, conflict_threshold, score_type, SCRIPT_NAME)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return


if __name__ == '__main__':
    submit_do_MWIS_runs()
