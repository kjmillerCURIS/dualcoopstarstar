import os
import sys


#PROMPT_MODE ${PROMPT_MODE}
#NUM_ENCODER_LAYERS ${NUM_ENCODER_LAYERS}
#NUM_DECODER_LAYERS ${NUM_DECODER_LAYERS}
#USE_POSITIONAL_EMBEDDING_IN_LOWER_PART ${SPATIAL_LOWER}
#USE_POSITIONAL_EMBEDDING_IN_UPPER_PART ${SPATIAL_UPPER}
#TEMPERATURE_IS_LEARNABLE ${TEMP_LEARNABLE}


DEBUG = False
SCRIPT_NAME = 'generic_COCO_dualcoopstarstar_fullpartial.sh'


def submit_dualcoopstarstar_fullpartial_runs():
    for prompt_mode in ['pos_only_fixed_prompt', 'pos_and_neg_learnable_prompt']:
        for (num_encoder_layers, num_decoder_layers) in [(0,1), (0,2), (2,2)]:
            for (spatial_lower, spatial_upper) in [(1,1)]: #[(0,0), (1,1)]:
                for temp_learnable in [0,1]:
                    for loss_averaging_mode in ['mean_except_class']:
                        job_name = 'dualcoopstarstar_fullpartial_coco_p05_seed1_%s_enc%d_dec%d_spatl%d_spatu%d_templearn%d_%s'%({'pos_only_fixed_prompt' : 'posonly_fixed', 'pos_and_neg_learnable_prompt' : 'posneglearn'}[prompt_mode], num_encoder_layers, num_decoder_layers, spatial_lower, spatial_upper, temp_learnable, loss_averaging_mode)
                        my_cmd = 'qsub -N %s -v run_ID=%s,PROMPT_MODE=%s,NUM_ENCODER_LAYERS=%d,NUM_DECODER_LAYERS=%d,SPATIAL_LOWER=%d,SPATIAL_UPPER=%d,TEMP_LEARNABLE=%d,LOSS_AVERAGING_MODE=%s %s'%(job_name, job_name, prompt_mode, num_encoder_layers, num_decoder_layers, spatial_lower, spatial_upper, temp_learnable, loss_averaging_mode, SCRIPT_NAME)
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_dualcoopstarstar_fullpartial_runs(*(sys.argv[1:]))
