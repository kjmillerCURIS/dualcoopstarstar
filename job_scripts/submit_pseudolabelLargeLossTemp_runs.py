import os
import sys


#cfg.TRAIN.PSEUDOLABEL_OBSERVATION_METHOD = 'observe_nothing' #or could be 'observe_positives'
#cfg.TRAIN.DELTA_REL = 0.04 #or could be 0.2 paired with max_epoch=9
#cfg.TRAIN.MAX_EPOCH_FOR_DELTA_REL = 49 #or could be 9 paired with delta_rel=0.2


DEBUG = False
SCRIPT_NAME = 'generic_COCO_pseudolabelLargeLossTemp.sh'


def submit_pseudolabelLargeLossTemp_runs():
    for use_bias in [0]:
        for observation_method in ['observe_positives']: #['observe_nothing', 'observe_positives']:
            for delta_rel, max_epoch in [(0.0, 0)]: #[(0.5, 9), (0.1, 49), (0.04, 49), (0.2, 9)]:
                job_name = 'pseudoLargeLossTemp_coco_seed1_bias%d_%s_deltaRel%.5f_maxEpochForDeltaRel%d'%(use_bias, observation_method, delta_rel, max_epoch)
                my_cmd = 'qsub -N %s -v run_ID=%s,USE_BIAS=%d,PSEUDOLABEL_OBSERVATION_METHOD=%s,DELTA_REL=%.5f,MAX_EPOCH_FOR_DELTA_REL=%d %s'%(job_name, job_name, use_bias, observation_method, delta_rel, max_epoch, SCRIPT_NAME)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return


if __name__ == '__main__':
    submit_pseudolabelLargeLossTemp_runs(*(sys.argv[1:]))
