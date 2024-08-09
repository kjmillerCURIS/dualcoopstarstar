import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_learn_logreg_weights_from_gt_stats.sh'


def submit_learn_logreg_weights_from_gt_stats_miniclass():
    for dataset_name in ['COCO2014_partial', 'nuswide_partial']:
        for input_type in ['cossims']:
            for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]:
                for matmul_by_XTXpinv in [0, 1]:
                    for append_input_stats in [0, 1]:
                        for isolate_diags in [0, 1]:
                            job_name = 'learn_logreg_weights_from_gt_stats_%s_%s_C%s_miniclass1_matmulXTXpinv%d_appendinputstats%d_isolatediags%d'%(dataset_name.split('_')[0], input_type, str(C), matmul_by_XTXpinv, append_input_stats, isolate_diags)
                            my_cmd = 'qsub -N %s -v DATASET_NAME=%s,INPUT_TYPE=%s,C=%s,MINICLASS=1,MATMUL_BY_XTXPINV=%d,APPEND_INPUT_STATS=%d,ISOLATE_DIAGS=%d %s'%(job_name, dataset_name, input_type, str(C), matmul_by_XTXpinv, append_input_stats, isolate_diags, SCRIPT_NAME)
                            print('submitting training run: "%s"'%(my_cmd))
                            os.system(my_cmd)
                            if DEBUG:
                                print('DEBUG MODE: let\'s see how that first run goes...')
                                return


if __name__ == '__main__':
    submit_learn_logreg_weights_from_gt_stats_miniclass()
