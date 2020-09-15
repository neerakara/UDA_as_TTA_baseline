#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch main.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
## SBATCH is recognized even though there is a # in front of it. If we want to comment SBATCH, we need ## in front of it. 
#SBATCH  --output=logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G

# activate virtual environment
# source /usr/bmicnas01/data-biwi-01/nkarani/softwares/anaconda/installation_dir/etc/profile.d/conda.sh
# conda activate tf_v1_12
source /usr/bmicnas01/data-biwi-01/nkarani/softwares/anaconda/installation_dir/bin/activate tf_v1_12

## EXECUTION OF PYTHON CODE:
python /usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/methods/baselines/uda/adv_feature_classifier/update_i2l_uda_invariant_features.py