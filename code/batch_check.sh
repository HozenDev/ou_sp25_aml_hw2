#!/bin/bash

# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=1G

# The %j is translated into the job number
#SBATCH --output=results/hw2_%j_stdout.txt
#SBATCH --error=results/hw2_%j_stderr.txt

#SBATCH --time=00:20:00
#SBATCH --job-name=hw2
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw2/code
#SBATCH --array=0-89
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Clean results repo and wandb
# ./clean.sh 

# Define experiment parameters
EXPERIMENT_TYPE='bmi'
DATASET='/home/fagg/datasets/bmi/bmi_dataset.pkl'
NTRAINING_VALUES=(1 2 3 4 6 8 11 14 18)
ROTATION=(0 2 4 6 8 10 12 14 16 18)
NTRAINING_LENGTH=${#NTRAINING_VALUES[@]}

NTRAINING_INDEX=$(($SLURM_ARRAY_TASK_ID % $NTRAINING_LENGTH))
ROTATION_INDEX=$(($SLURM_ARRAY_TASK_ID / $NTRAINING_LENGTH))

# --Ntraining ${NTRAINING_VALUES[$EXP_INDEX]} \  
# --Ntraining 14

# Using GPU add this to python execution
# --cpus-per-task $SLURM_CPUS_PER_TASK \

python hw2.py \
       --hidden 200 100 50 25 12 6 \
       --lrate 0.001 \
       --output_type dtheta \
       --predict_dim 1 \
       --epochs 1000 \
       --exp_type $EXPERIMENT_TYPE \
       --exp_index $SLURM_ARRAY_TASK_ID \
       --dataset $DATASET \
       --Ntraining ${NTRAINING_VALUES[$EXP_INDEX]} \
       --rotation $ROTATION \
       --activation_out 'linear' \
       --activation_hidden 'elu' \
       --label "exp" \
       --nowandb \
       --check \
