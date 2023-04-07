#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p gpu4_medium
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=without_bitfit
#SBATCH --mail-type=END
#SBATCH --output=outputs/train_without_bitfit.out
#SBATCH --gres=gpu:1

conda activate nyu-nlu
cd /gpfs/home/kgz2437/pathology_parsing/testing/NLU/Natural-Language-Understanding/hw3-main
python train_model.py False
