import os
import sys


DEBUG = False
SCRIPT_NAME = 'zsclip_llmensemble_COCO.sh'


def submit_zsclip_llmensemble_runs():
    for appendOrigClassname in ['_appendOrigClassname', '']:
        for betterPolysemy in ['_betterPolysemy', '']:
            for k in ['2', '4']: #['8', '16', '32']:
                llm_name = 'synonyms' + k + betterPolysemy + appendOrigClassname
                job_name = 'zsclip_llmensemble_COCO_' + llm_name
                my_cmd = 'qsub -N %s -v run_ID=%s,llm_name=%s %s'%(job_name, job_name, llm_name, SCRIPT_NAME)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return


if __name__ == '__main__':
    submit_zsclip_llmensemble_runs(*(sys.argv[1:]))
