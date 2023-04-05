#!/bin/bash
#SBATCH -p gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=bert-with-bitfit
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --gres=gpu:1

cd /gpfs/home/kgz2437/pathology_parsing/testing/NLU/Natural-Language-Understanding/hw3-main
test_model.py checkpoints/checkpoint.bitfit.04-02-2023.15-35-57
