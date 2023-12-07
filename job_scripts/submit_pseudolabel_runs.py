import os
import sys


#TRAINER.Caption.USE_BIAS ${USE_BIAS} \
#TRAIN.DO_ADJUST_LOGITS ${DO_ADJUST_LOGITS} \
#TRAIN.PSEUDOLABEL_UPDATE_GAUSSIAN_BANDWIDTH ${BANDWIDTH} \
#TRAIN.PSEUDOLABEL_UPDATE_STEPSIZE ${STEPSIZE}


DEBUG = False
SCRIPT_NAME = 'generic_COCO_pseudolabel.sh'


def submit_pseudolabel_runs():
    for do_adjust_logits in [1, 0]:
        for use_bias in [1, 0]:
            for bandwidth in [0.05, 0.1, 0.2, 0.4, 0.8]:
                for stepsize in [0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
                    if bandwidth in [0.1, 0.2, 0.4] and stepsize in [0.0625, 0.125, 0.25, 0.5, 1.0]:
                        continue
                    if stepsize == 0 and bandwidth != 0.2:
                        continue
                    if use_bias == 1 and (stepsize <= 1.0 or bandwidth in [0.1, 0.4]):
                        continue
                    job_name = 'pseudolabel_coco_seed1_adjust%d_bias%d_bandwidth%.5f_stepsize%.5f'%(do_adjust_logits, use_bias, bandwidth, stepsize)
                    my_cmd = 'qsub -N %s -v run_ID=%s,DO_ADJUST_LOGITS=%d,USE_BIAS=%d,BANDWIDTH=%.5f,STEPSIZE=%.5f %s'%(job_name, job_name, do_adjust_logits, use_bias, bandwidth, stepsize, SCRIPT_NAME)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return


if __name__ == '__main__':
    submit_pseudolabel_runs(*(sys.argv[1:]))
