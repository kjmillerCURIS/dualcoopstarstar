import os
import sys


#TRAIN.TERNARY_COOCCURRENCE_MAT_NAME ${TERNARY_COOCCURRENCE_MAT_NAME} \
#TRAIN.TERNARY_COOCCURRENCE_ALPHA ${TERNARY_COOCCURRENCE_ALPHA} \
#TRAIN.TERNARY_COOCCURRENCE_BETA_OVER_ALPHA ${TERNARY_COOCCURRENCE_BETA_OVER_ALPHA}


DEBUG = False
SCRIPT_NAME = 'generic_COCO_frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial.sh'


def submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_correctinitial():
    for ternary_cooccurrence_loss_type in ['correlation', 'prob_stopgrad_logit']:
        loss_type_str = ('' if ternary_cooccurrence_loss_type == 'prob_stopgrad_logit' else ternary_cooccurrence_loss_type + '_')
        for ternary_cooccurrence_mat_name in ['expectedsurprise_removepositives_20loc_1lpc']: #['gt_epsilon0.25_zeta0.25','expectedsurprise_confonly','expectedsurprise_fillin_20loc_3lpc','expectedsurprise_fillin_missingmiddle_20loc_3lpchigh_1lpclow']:
            for ternary_cooccurrence_alpha in {'correlation' : [0.0075], 'prob_stopgrad_logit' : [0.25]}[ternary_cooccurrence_loss_type]:
                for ternary_cooccurrence_beta_over_alpha in [0.5]:
                    job_name = 'frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_%s%s_%.5f_%.5f_coco_seed1'%(loss_type_str, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha)
                my_cmd = 'qsub -N  %s -v run_ID=%s,TERNARY_COOCCURRENCE_LOSS_TYPE=%s,TERNARY_COOCCURRENCE_MAT_NAME=%s,TERNARY_COOCCURRENCE_ALPHA=%.5f,TERNARY_COOCCURRENCE_BETA_OVER_ALPHA=%.5f %s'%(job_name, job_name, ternary_cooccurrence_loss_type, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha, SCRIPT_NAME)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_correctinitial()
