#!/bin/bash

# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu

#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=1G

# The %j is translated into the job number
#SBATCH --output=results/hw2_%j_stdout.txt
#SBATCH --error=results/hw2_%j_stderr.txt

#SBATCH --time=00:05:00
#SBATCH --job-name=hw2
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw2/code
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Clean results repo and wandb
# ./clean.sh 

python plot_fig.py
