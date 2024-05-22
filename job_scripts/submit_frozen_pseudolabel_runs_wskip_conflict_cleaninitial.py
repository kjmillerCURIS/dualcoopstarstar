import os
import sys


#TRAIN.TERNARY_COOCCURRENCE_MAT_NAME ${TERNARY_COOCCURRENCE_MAT_NAME} \
#TRAIN.TERNARY_COOCCURRENCE_ALPHA ${TERNARY_COOCCURRENCE_ALPHA} \
#TRAIN.TERNARY_COOCCURRENCE_BETA_OVER_ALPHA ${TERNARY_COOCCURRENCE_BETA_OVER_ALPHA}


DEBUG = True
SCRIPT_NAME = 'generic_COCO_frozen_pseudolabel_wskip_conflict_cleaninitial.sh'


def submit_frozen_pseudolabel_runs_wskip_conflict_cleaninitial():
    for conflict_threshold in [0.001, 0.2]:
        conflict_matrix_name = 'gt_conflict_matrix_threshold%f'%(conflict_threshold)
        for clean_p in [0.01, 0.1]:
            for clean_q in [0.1, 0.4]:
                for clean_r in [0.1, 0.02]:
                    job_name = 'frozen_pseudolabel_wskip_conflict_cleaninitial_%s_p%.5f_q%.5f_r%.5f_coco_seed1'%(conflict_matrix_name, clean_p, clean_q, clean_r)
                    my_cmd = 'qsub -N  %s -v run_ID=%s,CONFLICT_MATRIX_NAME=%s,CLEAN_P=%f,CLEAN_Q=%f,CLEAN_R=%f %s'%(job_name, job_name, conflict_matrix_name, clean_p, clean_q, clean_r, SCRIPT_NAME)
                    print('submitting training run: "%s"'%(my_cmd))
                    os.system(my_cmd)
                    if DEBUG:
                        print('DEBUG MODE: let\'s see how that first run goes...')
                        return


if __name__ == '__main__':
    submit_frozen_pseudolabel_runs_wskip_conflict_cleaninitial()
