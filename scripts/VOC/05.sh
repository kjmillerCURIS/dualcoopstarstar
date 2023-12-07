#!/bin/bash -l

#$ -N VOC_05

#$ -m bea

#$ -M pinghu@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Request 4 CPUs
#$ -pe omp 2

# Request 2 GPU
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=7.5

#$ -l gpu_memory=48G


#$ -l h_rt=40:00:00

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load miniconda
module load cuda/11.1
module load gcc
conda activate py37
#conda install -c conda-forge opencv
export PROJECT_PATH=/projectnb/ivc-ml


cd $PROJECT_PATH/pinghu/proj/TaI-DPT/


# custom config
DATA=$PROJECT_PATH/pinghu/dataset
TRAINER=Caption_tri_wta_soft

DATASET=voc2007_partial
CFG=rn101_bn96  # config file
CTP=end  # class token position (end or middle)
NCTX=21  # number of context tokens
CSC=True  # class-specific context (False or True)
MOD=pos200
run_ID=voc2007_partial_tricoop_wta_soft_448_CSC_p0_5-pos200-ctx21
partial_prob=0.5

for SEED in 1 3 5
do
    DIR=exp_voc/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job andsave the output to ${DIR}"
        python train_caption.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --mode ${MOD} \
        TRAINER.Caption.N_CTX ${NCTX} \
        TRAINER.Caption.CSC ${CSC} \
        TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.partial_prob ${partial_prob}
    fi
done


run_ID=voc2007_partial_tricoop_wta_soft_448_CSC_p0_6-pos200-ctx21
partial_prob=0.6

for SEED in 1 3 5
do
    DIR=exp_voc/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job andsave the output to ${DIR}"
        python train_caption.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --mode ${MOD} \
        TRAINER.Caption.N_CTX ${NCTX} \
        TRAINER.Caption.CSC ${CSC} \
        TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.partial_prob ${partial_prob}
    fi
done
