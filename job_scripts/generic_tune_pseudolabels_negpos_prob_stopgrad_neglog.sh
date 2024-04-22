#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=16G
#$ -l h_rt=1:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/tune_pseudolabels_negpos_prob_stopgrad_neglog.py ${ALPHA} ${EPSILON} ${BETA} ${ZETA}

