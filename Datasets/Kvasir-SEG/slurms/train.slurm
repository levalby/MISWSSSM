#!/bin/bash
#SBATCH -J train
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t hh:mm:ss
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 40G
#SBATCH --gres=gpu:a40:1
source ~/.bashrc
conda activate MODEL_ENV
cd MODEL
bash ./train_n_evaluate_kvasir-seg.sh MODEL_NAME