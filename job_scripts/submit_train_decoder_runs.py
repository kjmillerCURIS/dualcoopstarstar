import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_train_decoder.sh'


def submit_train_decoder_runs():
    for dataset_name in ['nuswide_partial']: #['COCO2014_partial']:
        for use_dropout, use_batchnorm in zip([0, 1, 1, 0], [0, 1, 0, 1]):
            for input_type in ['cossims', 'logits']:
                for hidden_layer_size in [16, 128, 1024]:
                    for num_hidden_layers in [1,2,3]:
                        for lr in [5e-3, 5e-2, 5e-4]:
                            job_name = 'train_decoder_%s_%s_nhl%d_hls%d_dropout%d_batchnorm%d_lr%s'%(dataset_name.split('_')[0], input_type, num_hidden_layers, hidden_layer_size, use_dropout, use_batchnorm, str(lr))
                            my_cmd = 'qsub -N %s -v DATASET_NAME=%s,INPUT_TYPE=%s,NUM_HIDDEN_LAYERS=%d,HIDDEN_LAYER_SIZE=%d,USE_DROPOUT=%d,USE_BATCHNORM=%d,LR=%s %s'%(job_name, dataset_name, input_type, num_hidden_layers, hidden_layer_size, use_dropout, use_batchnorm, str(lr), SCRIPT_NAME)
                            os.system(my_cmd)
                            if DEBUG:
                                print('DEBUG MODE: let\'s see how that first run goes...')
                                return


if __name__ == '__main__':
    submit_train_decoder_runs()
