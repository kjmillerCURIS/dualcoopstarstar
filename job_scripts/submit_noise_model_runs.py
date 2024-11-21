import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_noise_model.sh'
MODEL_TYPE_LIST = ['RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-L14', 'RN50', 'ViT-B16', 'ViT-L14336px']
DATASET_NAME_LIST = ['nuswideTheirVersion_partial'] #['VOC2007_partial', 'COCO2014_partial', 'nuswideTheirVersion_partial']


def submit_noise_model_runs():
    for model_type in MODEL_TYPE_LIST:
        for dataset_name in DATASET_NAME_LIST:
            job_name = 'noise_model_%s_%s'%(dataset_name.split('_')[0], model_type)
            my_cmd = 'qsub -N %s -v DATASET_NAME=%s,MODEL_TYPE=%s %s'%(job_name, dataset_name, model_type, SCRIPT_NAME)
            print('submitting training run: "%s"'%(my_cmd))
            os.system(my_cmd)
            if DEBUG:
                print('DEBUG MODE: let\'s see how that first run goes...')
                return


if __name__ == '__main__':
    submit_noise_model_runs()
