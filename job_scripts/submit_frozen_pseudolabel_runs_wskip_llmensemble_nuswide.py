import os
import sys


#llm_name ${llm_name} \
#TRAINER.Caption.THREE_SEPARATE_ENSEMBLES ${THREE_SEPARATE_ENSEMBLES} \
#TRAIN.INIT_WITH_ORIG_CLASSNAMES_ONLY ${ORIG_CLASSNAMES_INIT}


DEBUG = False
SCRIPT_NAME = 'generic_nuswide_frozen_pseudolabel_wskip_learnedensemble.sh'


def submit_frozen_pseudolabel_runs_wskip_llmensemble_nuswide():
    for num_synonyms in ['8']:
        for appendClass in ['_appendOrigClassname']:
            for betterPolysemy in ['', '_betterPolysemy']:
                llm_name = 'synonyms%s%s%s'%(num_synonyms, betterPolysemy, appendClass)
                for orig_classnames_init in [1]:
                    for three_sep in [0]:
                        job_name = 'frozen_pseudolabel_wskip_%s_originit%d_threesep%d_nuswide_seed1'%(llm_name, orig_classnames_init, three_sep)
                        my_cmd = 'qsub -N %s -v run_ID=%s,llm_name=%s,ORIG_CLASSNAMES_INIT=%d,THREE_SEPARATE_ENSEMBLES=%d %s'%(job_name, job_name, llm_name, orig_classnames_init, three_sep, SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_wskip_llmensemble_nuswide()
