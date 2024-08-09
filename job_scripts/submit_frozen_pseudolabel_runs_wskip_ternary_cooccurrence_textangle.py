import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_COCO_frozen_pseudolabel_wskip_ternary_cooccurrence_textangle_lod.sh'


def submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_textangle():
    for ternary_cooccurrence_loss_off_duration in [0]:
            for ternary_cooccurrence_mat_name in ['gt_epsilon0.25_zeta0.25']: #['gt_epsilon0.25_zeta0.25','expectedsurprise_confonly','expectedsurprise_fillin_20loc_3lpc']:
                for ternary_cooccurrence_alpha in [1e-2, 1e-5, 1e-4, 1e-3, 1e-1, 1.0]:
                    for ternary_cooccurrence_beta_over_alpha in [1.0, 0.0, 0.25, 0.5, 0.75, 1.25, 1.5]:
                        job_name = 'frozen_pseudolabel_wskip_ternary_cooccurrence_textangle_lod%d_%s_%s_%s_coco_seed1'%(ternary_cooccurrence_loss_off_duration, ternary_cooccurrence_mat_name, str(ternary_cooccurrence_alpha), str(ternary_cooccurrence_beta_over_alpha))
                        my_cmd = 'qsub -N  %s -v run_ID=%s,TERNARY_COOCCURRENCE_LOSS_OFF_DURATION=%d,TERNARY_COOCCURRENCE_MAT_NAME=%s,TERNARY_COOCCURRENCE_ALPHA=%s,TERNARY_COOCCURRENCE_BETA_OVER_ALPHA=%s %s'%(job_name, job_name, ternary_cooccurrence_loss_off_duration, ternary_cooccurrence_mat_name, str(ternary_cooccurrence_alpha), str(ternary_cooccurrence_beta_over_alpha), SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_textangle()
