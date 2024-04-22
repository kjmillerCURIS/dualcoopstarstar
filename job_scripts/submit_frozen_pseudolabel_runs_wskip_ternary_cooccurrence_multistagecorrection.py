import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_COCO_frozen_pseudolabel_wskip_ternary_cooccurrence_multistagecorrection.sh'


def submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_multistagecorrection():
    for ternary_cooccurrence_loss_type in ['prob_stopgrad_logit', 'correlation']:
        for ternary_cooccurrence_mat_name in ['gt_epsilon0.25_zeta0.25']: #['gt_epsilon0.25_zeta0.25','expectedsurprise_confonly','expectedsurprise_fillin_20loc_3lpc','expectedsurprise_fillin_missingmiddle_20loc_3lpchigh_1lpclow']:
            for ternary_cooccurrence_alpha in {'prob_stopgrad_logit' : [0.25] , 'correlation' : [0.0075]}[ternary_cooccurrence_loss_type]:
                for ternary_cooccurrence_beta_over_alpha in [0.5]:
                    for multistagecorrection_epochs in ['_3_', '_5_', '_9_']:
                        for multistagecorrection_mu in [0.0, 0.5, 0.9]:
                            job_name = 'frozen_pseudolabel_wskip_ternary_cooccurrence_multistagecorrection_%s_%s_%.5f_%.5f_-%s-_%.5f_coco_seed1'%(ternary_cooccurrence_loss_type, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha, multistagecorrection_epochs.strip('_'), multistagecorrection_mu)
                            my_cmd = 'qsub -N  %s -v run_ID=%s,TERNARY_COOCCURRENCE_LOSS_TYPE=%s,TERNARY_COOCCURRENCE_MAT_NAME=%s,TERNARY_COOCCURRENCE_ALPHA=%.5f,TERNARY_COOCCURRENCE_BETA_OVER_ALPHA=%.5f,MULTISTAGECORRECTION_EPOCHS=%s,MULTISTAGECORRECTION_MU=%.5f %s'%(job_name, job_name, ternary_cooccurrence_loss_type, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha, multistagecorrection_epochs, multistagecorrection_mu, SCRIPT_NAME)
                            print('submitting training run: "%s"'%(my_cmd))
                            os.system(my_cmd)
                            if DEBUG:
                                print('DEBUG MODE: let\'s see how that first run goes...')
                                return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_wskip_ternary_cooccurrence_multistagecorrection()
