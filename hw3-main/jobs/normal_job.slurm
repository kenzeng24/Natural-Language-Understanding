#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p gpu4_medium
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=bert-bitfit
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --gres=gpu:1

cd /gpfs/home/kgz2437/pathology_parsing/testing/NLU/Natural-Language-Understanding/hw3-main
python train_model.py False
