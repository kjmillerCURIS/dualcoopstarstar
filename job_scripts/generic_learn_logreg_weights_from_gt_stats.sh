#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/learn_logreg_weights_from_gt_stats.py ${DATASET_NAME} ${INPUT_TYPE} ${C} ${MINICLASS} ${MATMUL_BY_XTXPINV} ${APPEND_INPUT_STATS} ${ISOLATE_DIAGS}

