#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=2:59:59
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 2
#$ -j y
#$ -m ea

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/noise_model.py ${DATASET_NAME} ${MODEL_TYPE}

