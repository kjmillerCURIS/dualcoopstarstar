import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_compute_cossims_train_noaug.sh'
MODEL_TYPE_LIST = ['RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14', 'ViT-L14336px']
SINGLE_PROBE_TYPE_LIST = ['classname_only', 'single_standard', 'ensemble_80']


def submit_compute_cossims_train_noaug_runs():
    for dataset_name in ['nuswide_partial', 'VOC2007_partial']: #['COCO2014_partial']:
        for model_type in ['ViT-L14336px']: #MODEL_TYPE_LIST:
            for single_probe_type in ['ensemble_80']: #SINGLE_PROBE_TYPE_LIST:
                job_name = 'compute_cossims_train_noaug_%s_%s_%s'%(dataset_name, model_type, single_probe_type)
                my_cmd = 'qsub -N %s -v DATASET_NAME=%s,MODEL_TYPE=%s,SINGLE_PROBE_TYPE=%s %s'%(job_name, dataset_name, model_type, single_probe_type, SCRIPT_NAME)
                print('submitting training run: "%s"'%(my_cmd))
                os.system(my_cmd)
                if DEBUG:
                    print('DEBUG MODE: let\'s see how that first run goes...')
                    return


if __name__ == '__main__':
    submit_compute_cossims_train_noaug_runs()
