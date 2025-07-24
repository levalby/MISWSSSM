#!/bin/bash
#SBATCH -J trainUmamba
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t hh:mm:ss
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 40G
#SBATCH --gres=gpu:a40:1
source ~/.bashrc
conda activate umamba
cd U-Mamba
bash ./scripts/train_SCRIPT_NAME.sh MODEL_NAME DIMENSION NUM_GPUS 