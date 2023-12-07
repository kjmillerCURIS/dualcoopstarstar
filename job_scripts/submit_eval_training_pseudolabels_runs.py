import os
import sys


#TRAINER.Caption.USE_BIAS ${USE_BIAS} \
#TRAIN.DO_ADJUST_LOGITS ${DO_ADJUST_LOGITS} \
#TRAIN.PSEUDOLABEL_UPDATE_GAUSSIAN_BANDWIDTH ${BANDWIDTH} \
#TRAIN.PSEUDOLABEL_UPDATE_STEPSIZE ${STEPSIZE}


DEBUG = False
SCRIPT_NAME = 'eval_training_pseudolabels_generic_COCO.sh'


def submit_eval_training_pseudolabels_runs():
    for do_adjust_logits in [1]: #[1, 0]:
        for use_bias in [1, 0]:
            for bandwidth in [0.1, 0.2, 0.4]:
                for stepsize in [0.0625, 0.125, 0.25, 0.5, 1.0]:
                    job_name = 'pseudolabel_coco_seed1_adjust%d_bias%d_bandwidth%.5f_stepsize%.5f'%(do_adjust_logits, use_bias, bandwidth, stepsize)
                    output_dir = os.path.join('../../vislang-domain-exploration-data/dualcoopstarstar-data/output', job_name)
                    if not os.path.exists(output_dir):
                        print('not submitting job because dir "%s" does not exist'%(output_dir))
                        continue

                    my_cmd = 'qsub -N %s -v run_ID=%s %s'%(job_name, job_name, SCRIPT_NAME)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return


if __name__ == '__main__':
    submit_eval_training_pseudolabels_runs(*(sys.argv[1:]))
