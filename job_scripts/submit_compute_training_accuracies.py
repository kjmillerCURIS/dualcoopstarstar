import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_compute_training_accuracies.sh'
DATA_BASE = '../vislang-domain-exploration-data/dualcoopstarstar-data/output'


def submit_compute_training_accuracies():
    job_list = []

    #baseline
    job_list.append(('frozen_pseudolabel_with_skipping_coco_seed1', 'baseline', 0))

    #grad descent
    for lod in [0,1,3]:
        job_list.append(('frozen_pseudolabel_wskip_ternary_cooccurrence_lod%d_prob_stopgrad_logit_gt_epsilon0.25_zeta0.25_0.00200_0.75000_coco_seed1'%(lod), 'coocgrad-0.002_0.75_lod%d'%(lod), 0))

    job_list.append(('frozen_pseudolabel_wskip_ternary_cooccurrence_lod3_prob_stopgrad_logit_gt_epsilon0.25_zeta0.25_0.12500_0.75000_coco_seed1', 'coocgrad-0.125_0.75_lod3', 0))

    #correctinitial
    job_list.append(('frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_gt_epsilon0.25_zeta0.25_0.25000_0.50000_coco_seed1', 'correctinitial-prob_stopgrad_logit', 1))
    job_list.append(('frozen_pseudolabel_wskip_ternary_cooccurrence_correctinitial_correlation_gt_epsilon0.25_zeta0.25_0.00750_0.50000_coco_seed1', 'correctinitial-correlation', 1))

    #multistage
    for a in [3, 9]:
        for b in [0.0, 0.9]:
            for c in ['correlation', 'prob_stopgrad_logit']:
                job_list.append(('frozen_pseudolabel_wskip_ternary_cooccurrence_multistagecorrection_%s_gt_epsilon0.25_zeta0.25_%.5f_0.50000_-%d-_%.5f_coco_seed1'%(c, {'prob_stopgrad_logit' : 0.25, 'correlation' : 0.0075}[c], a, b), 'multistagecorrection-%s-epoch%d-mu%.5f'%(c, a, b), 2))

    job_list.reverse()
    for job_tuple in job_list:
        my_cmd = 'qsub -N cta_%s -v JOB_DIR=%s,TITLE=%s,CORRECTION_LEVEL=%d %s'%(job_tuple[1], os.path.join(DATA_BASE, job_tuple[0]), job_tuple[1], job_tuple[2], SCRIPT_NAME)
        print('submitting training run: "%s"'%(my_cmd))
        os.system(my_cmd)
        if DEBUG:
            print('DEBUG MODE: let\'s see how that first run goes...')
            return


if __name__ == '__main__':
    submit_compute_training_accuracies()
