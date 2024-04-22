import os
import sys


#llm_name ${llm_name} \
#TRAINER.Caption.THREE_SEPARATE_ENSEMBLES ${THREE_SEPARATE_ENSEMBLES} \
#TRAIN.INIT_WITH_ORIG_CLASSNAMES_ONLY ${ORIG_CLASSNAMES_INIT}
#TRAIN.TERNARY_COOCCURRENCE_LOSS_TYPE ${TERNARY_COOCCURRENCE_LOSS_TYPE} \
#TRAIN.TERNARY_COOCCURRENCE_MAT_NAME ${TERNARY_COOCCURRENCE_MAT_NAME} \
#TRAIN.TERNARY_COOCCURRENCE_ALPHA ${TERNARY_COOCCURRENCE_ALPHA} \
#TRAIN.TERNARY_COOCCURRENCE_BETA_OVER_ALPHA ${TERNARY_COOCCURRENCE_BETA_OVER_ALPHA} \


DEBUG = False
SCRIPT_NAME = 'generic_voc2007_frozen_pseudolabel_wskip_learnedensemble_ternary_cooccurrence_correctinitial.sh'


def submit_frozen_pseudolabel_runs_wskip_llmensemble_ternary_cooccurrence_correctinitial_voc2007():
    for ternary_cooccurrence_loss_type in ['prob_stopgrad_logit', 'correlation']:
        for ternary_cooccurrence_mat_name in ['gt_epsilon0.25_zeta0.25','expectedsurprise_removepositives_20loc_1lpc','expectedsurprise_confonly','expectedsurprise_fillin_20loc_3lpc','expectedsurprise_fillin_missingmiddle_20loc_3lpchigh_1lpclow']:
            for ternary_cooccurrence_alpha in {'prob_stopgrad_logit' : [0.25], 'correlation' : [0.0075]}[ternary_cooccurrence_loss_type]:
                for ternary_cooccurrence_beta_over_alpha in {'prob_stopgrad_logit' : [0.5], 'correlation' : [0.5]}[ternary_cooccurrence_loss_type]:
                    for num_synonyms in ['8']:
                        for appendClass in ['_appendOrigClassname']: #['_appendOrigClassname', '']:
                            for betterPolysemy in ['_betterPolysemy', '']: #['', '_betterPolysemy']:
                                llm_name = 'synonyms%s%s%s'%(num_synonyms, betterPolysemy, appendClass)
                                for orig_classnames_init in [1]: #[1, 0]:
                                    for three_sep in [0]: #[1, 0]:
                                        job_name = 'frozen_pseudolabel_wskip_%s_originit%d_threesep%d_ternary_cooccurrence_correctinitial_%s_%s_%.5f_%.5f_voc2007_seed1'%(llm_name, orig_classnames_init, three_sep, ternary_cooccurrence_loss_type, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha)
                                        my_cmd = 'qsub -N %s -v run_ID=%s,llm_name=%s,ORIG_CLASSNAMES_INIT=%d,THREE_SEPARATE_ENSEMBLES=%d,TERNARY_COOCCURRENCE_LOSS_TYPE=%s,TERNARY_COOCCURRENCE_MAT_NAME=%s,TERNARY_COOCCURRENCE_ALPHA=%.5f,TERNARY_COOCCURRENCE_BETA_OVER_ALPHA=%.5f %s'%(job_name, job_name, llm_name, orig_classnames_init, three_sep, ternary_cooccurrence_loss_type, ternary_cooccurrence_mat_name, ternary_cooccurrence_alpha, ternary_cooccurrence_beta_over_alpha, SCRIPT_NAME)
                                        print('submitting training run: "%s"'%(my_cmd))
                                        os.system(my_cmd)
                                        if DEBUG:
                                            print('DEBUG MODE: let\'s see how that first run goes...')
                                            return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_wskip_llmensemble_ternary_cooccurrence_correctinitial_voc2007()
