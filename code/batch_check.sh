#!/bin/bash

# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=1G

# The %j is translated into the job number
#SBATCH --output=results/hw2_check_%j_stdout.txt
#SBATCH --error=results/hw2_check_%j_stderr.txt

#SBATCH --time=00:20:00
#SBATCH --job-name=hw2_check
#SBATCH --chdir=/home/cs504305/hw2/code
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Clean results repo and wandb
# ./clean.sh 

python hw2.py --check --exp_type 'bmi' --label 'exp' --exp_type 'theta'
