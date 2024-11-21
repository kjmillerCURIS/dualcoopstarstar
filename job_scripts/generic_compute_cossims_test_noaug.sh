#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=5:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/compute_cossims_test_noaug.py ${DATASET_NAME} ${MODEL_TYPE} ${SINGLE_PROBE_TYPE}

