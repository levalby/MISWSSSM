#!/bin/bash
#SBATCH -J preprocessSwinUMamba
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 15G
#SBATCH --gres=gpu:a40:1
source ~/.bashrc
conda activate swin_umamba
cd Swin-UMamba/data
nnUNetv2_plan_and_preprocess -np 2 -d DATASET_ID --verify_dataset_integrity