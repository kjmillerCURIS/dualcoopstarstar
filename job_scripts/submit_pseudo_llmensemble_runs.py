import os
import sys



#llm_name ${llm_name} \
#TRAINER.Caption.THREE_SEPARATE_ENSEMBLES ${THREE_SEPARATE_ENSEMBLES} \
#TRAIN.INIT_WITH_ORIG_CLASSNAMES_ONLY ${ORIG_CLASSNAMES_INIT}


DEBUG = False
SCRIPT_NAME = {}
SCRIPT_NAME['CDUL1'] = 'generic_COCO_pseudolabel_learnedensemble.sh'
SCRIPT_NAME['frozen'] = 'generic_COCO_pseudolabel_learnedensemble.sh'
SCRIPT_NAME['LargeLoss1'] = 'generic_COCO_pseudolabelLargeLossTemp_learnedensemble.sh'
SCRIPT_NAME['AN'] = 'generic_COCO_pseudolabelLargeLossTemp_learnedensemble.sh'

COMMON_ARGS1 = ['USE_BIAS=0']
ALL_ARGS = {}
ALL_ARGS['CDUL1'] = COMMON_ARGS1 + ['DO_ADJUST_LOGITS=0', 'BANDWIDTH=%.5f'%(0.4), 'STEPSIZE=%.5f'%(0.25)]
ALL_ARGS['frozen'] = COMMON_ARGS1 + ['DO_ADJUST_LOGITS=0', 'BANDWIDTH=%.5f'%(0.2), 'STEPSIZE=%.5f'%(0.0)]
ALL_ARGS['LargeLoss1'] = COMMON_ARGS1 + ['PSEUDOLABEL_OBSERVATION_METHOD=%s'%('observe_positives'), 'DELTA_REL=%.5f'%(0.5), 'MAX_EPOCH_FOR_DELTA_REL=%d'%(9)]
ALL_ARGS['AN'] = COMMON_ARGS1 + ['PSEUDOLABEL_OBSERVATION_METHOD=%s'%('observe_positives'), 'DELTA_REL=%.5f'%(0.0), 'MAX_EPOCH_FOR_DELTA_REL=%d'%(0)]


def submit_pseudo_llmensemble_runs():
    for num_synonyms in ['8']:
        for pseudo_type in ['AN']: #['LargeLoss1', 'frozen', 'CDUL1']:
            for appendClass in ['_appendOrigClassname', '']:
                for orig_classnames_init in [1, 0]:
                    for betterPolysemy in ['_betterPolysemy', '']:
                        for three_sep in [1, 0]:
                            llm_name = 'synonyms' + num_synonyms + betterPolysemy + appendClass
                            script_name = SCRIPT_NAME[pseudo_type]
                            job_name = 'pseudo_llmensemble_coco_seed1_%s_%s_orig_classnames_init%d_three_sep%d'%(pseudo_type, llm_name, orig_classnames_init, three_sep)
                            all_args = ALL_ARGS[pseudo_type]
                            llm_args = ['llm_name=%s'%(llm_name), 'ORIG_CLASSNAMES_INIT=%d'%(orig_classnames_init), 'THREE_SEPARATE_ENSEMBLES=%d'%(three_sep)]
                            my_cmd = 'qsub -N %s -v run_ID=%s,'%(job_name, job_name) + ','.join(all_args + llm_args) + ' ' + script_name
                            print('submitting training run: "%s"'%(my_cmd))
                            os.system(my_cmd)
                            if DEBUG:
                                print('DEBUG MODE: let\'s see how that first run goes...')
                                return


if __name__ == '__main__':
    submit_pseudo_llmensemble_runs(*(sys.argv[1:]))
