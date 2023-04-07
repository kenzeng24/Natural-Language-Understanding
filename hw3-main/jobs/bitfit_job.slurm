#!/bin/bash
#SBATCH -p gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=with-bitfit
#SBATCH --mail-type=END
#SBATCH --output=outputs/train_with_bitfit.out
#SBATCH --gres=gpu:1

conda activaten nyu-nlu
cd /gpfs/home/kgz2437/pathology_parsing/testing/NLU/Natural-Language-Understanding/hw3-main
python train_model.py True
