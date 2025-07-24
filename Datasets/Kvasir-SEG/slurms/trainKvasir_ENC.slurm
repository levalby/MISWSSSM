#!/bin/bash
#SBATCH -J train
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 40G
#SBATCH --gres=gpu:a40:1
source ~/.bashrc
conda activate umamba
cd U-Mamba
bash ./train_n_evaluate_kvasir-seg.sh nnUNetTrainerUMambaEnc
