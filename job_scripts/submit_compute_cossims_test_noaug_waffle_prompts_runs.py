import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_compute_cossims_test_noaug_waffle_prompts.sh'
MODEL_TYPE_LIST = ['RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14', 'ViT-L14336px']


def submit_compute_cossims_test_noaug_waffle_prompts_runs():
    for dataset_name in ['VOC2007_partial', 'COCO2014_partial', 'nuswideTheirVersion_partial']:
        for model_type in MODEL_TYPE_LIST:
            job_name = 'compute_cossims_test_noaug_waffle_prompts_%s_%s'%(dataset_name, model_type)
            my_cmd = 'qsub -N %s -v DATASET_NAME=%s,MODEL_TYPE=%s %s'%(job_name, dataset_name, model_type, SCRIPT_NAME)
            print('submitting training run: "%s"'%(my_cmd))
            os.system(my_cmd)
            if DEBUG:
                print('DEBUG MODE: let\'s see how that first run goes...')
                return


if __name__ == '__main__':
    submit_compute_cossims_test_noaug_waffle_prompts_runs()
