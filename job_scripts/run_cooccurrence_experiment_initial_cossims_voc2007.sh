#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=1:59:59
#$ -j y
#$ -m ea
#$ -N init_cossims_voc2007

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/compute_initial_cossims.py VOC2007_partial

