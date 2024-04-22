import os
import sys


#TRAIN.TERNARY_COOCCURRENCE_LOSS_TYPE ${TERNARY_COOCCURRENCE_LOSS_TYPE} \
#TRAIN.TERNARY_COOCCURRENCE_MAT_NAME ${TERNARY_COOCCURRENCE_MAT_NAME} \
#TRAIN.TERNARY_COOCCURRENCE_ALPHA ${TERNARY_COOCCURRENCE_ALPHA} \
#TRAIN.TERNARY_COOCCURRENCE_BETA_OVER_ALPHA ${TERNARY_COOCCURRENCE_BETA_OVER_ALPHA}


DEBUG = False
#SCRIPT_NAME = 'generic_COCO_frozen_pseudolabel_wskip_ternary_cooccurrence.sh'
SCRIPT_NAME = 'generic_COCO_frozen_pseudolabel_wskip_ternary_cooccurrence_lod_use_gt_labels.sh'


def submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_use_gt_labels():
    for ternary_cooccurrence_loss_off_duration in [0, 1, 2, 3]:
        for ternary_cooccurrence_loss_type in ['prob_stopgrad_logit']:
            for ternary_cooccurrence_mat_name in ['gt_epsilon0.25_zeta0.25']: #['gt_epsilon0.25_zeta0.25','expectedsurprise_confonly','expectedsurprise_fillin_20loc_3lpc']:
                for ternary_cooccurrence_alpha in [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 0.0625, 0.125, 0.25]:
                    for ternary_cooccurrence_beta_over_alpha in [0.0]: #[0.0, 0.25, 0.5, 0.75]:
                        if ternary_cooccurrence_alpha == 0.0 and not (ternary_cooccurrence_loss_off_duration == 0 and ternary_cooccurrence_beta_over_alpha == 0):
                            continue

                        job_name = 'frozen_pseudolabel_wskip_ternary_cooccurrence_usegtlabels_lod%d_%s_%s_%.5f_%.5f_coco_seed1'%(ternary_cooccurrence_loss_off_duration, ternary_cooccurrence_loss_type, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha)
                        my_cmd = 'qsub -N  %s -v run_ID=%s,TERNARY_COOCCURRENCE_LOSS_OFF_DURATION=%d,TERNARY_COOCCURRENCE_LOSS_TYPE=%s,TERNARY_COOCCURRENCE_MAT_NAME=%s,TERNARY_COOCCURRENCE_ALPHA=%.5f,TERNARY_COOCCURRENCE_BETA_OVER_ALPHA=%.5f %s'%(job_name, job_name, ternary_cooccurrence_loss_off_duration, ternary_cooccurrence_loss_type, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha, SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_use_gt_labels()
