#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=1:59:59
#$ -j y
#$ -m ea
#$ -N init_pseudolabels_voc2007

module load miniconda
conda activate dualcoopstarstar
cd ~/data/dualcoopstarstar
python cooccurrence_correction_experiments/compute_initial_pseudolabels.py VOC2007_partial

