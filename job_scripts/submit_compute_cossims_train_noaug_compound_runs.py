import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_compute_cossims_train_noaug_compound.sh'
MODEL_TYPE_LIST = ['RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B32', 'ViT-B16', 'ViT-L14', 'ViT-L14336px']
SINGLE_PROBE_TYPE_LIST = ['ensemble_80']
COMPOUND_PROBE_TYPE_LIST = ['a_photo_of_a_i_and_a_j']


def submit_compute_cossims_train_noaug_compound_runs():
    for dataset_name in ['COCO2014_partial']:
        for model_type in MODEL_TYPE_LIST:
            for single_probe_type in SINGLE_PROBE_TYPE_LIST:
                for compound_probe_type in COMPOUND_PROBE_TYPE_LIST:
                    job_name = 'compute_cossims_train_noaug_compound_%s-%s-%s-%s'%(dataset_name, model_type, single_probe_type, compound_probe_type)
                    my_cmd = 'qsub -N %s -v DATASET_NAME=%s,MODEL_TYPE=%s,SINGLE_PROBE_TYPE=%s,COMPOUND_PROBE_TYPE=%s %s'%(job_name, dataset_name, model_type, single_probe_type, compound_probe_type, SCRIPT_NAME)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return


if __name__ == '__main__':
    submit_compute_cossims_train_noaug_compound_runs()
