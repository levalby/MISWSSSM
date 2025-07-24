#!/bin/bash
#SBATCH -J preprocessUMamba
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 15G
#SBATCH --gres=gpu:a40:1
source ~/.bashrc
conda activate umamba
cd U-Mamba/data
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity