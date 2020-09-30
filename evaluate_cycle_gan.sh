#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

# activate virtual environment
source /usr/bmicnas01/data-biwi-01/nkarani/softwares/anaconda/installation_dir/bin/activate tf_v1_12

## EXECUTION OF PYTHON CODE:
python /usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/methods/baselines/uda/adv_feature_classifier/evaluate_cycle_gan.py