#!/bin/bash
#SBATCH -J preprocessHarDNet-MSEG
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 15G
#SBATCH --gres=gpu:a40:1
source ~/.bashrc
conda activate MODEL_ENV
cd MODEL/data
nnUNetv2_plan_and_preprocess -np 2 -d 002 --verify_dataset_integrity