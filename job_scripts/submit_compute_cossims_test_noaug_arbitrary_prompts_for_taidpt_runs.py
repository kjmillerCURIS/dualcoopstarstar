import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_compute_cossims_test_noaug_arbitrary_prompts_for_taidpt.sh'
MODEL_TYPE_LIST = ['RN50']


def submit_compute_cossims_test_noaug_arbitrary_prompts_for_taidpt_runs():
    for dataset_name in ['nuswideTheirVersion_partial', 'COCO2014_partial']: #['VOC2007_partial', 'COCO2014_partial', 'nuswideTheirVersion_partial']:
        for model_type in MODEL_TYPE_LIST:
            job_name = 'for_taidpt_compute_cossims_test_noaug_arbitrary_prompts_%s_%s'%(dataset_name, model_type)
            my_cmd = 'qsub -N %s -v DATASET_NAME=%s,MODEL_TYPE=%s %s'%(job_name, dataset_name, model_type, SCRIPT_NAME)
            print('submitting training run: "%s"'%(my_cmd))
            os.system(my_cmd)
            if DEBUG:
                print('DEBUG MODE: let\'s see how that first run goes...')
                return


if __name__ == '__main__':
    submit_compute_cossims_test_noaug_arbitrary_prompts_for_taidpt_runs()
