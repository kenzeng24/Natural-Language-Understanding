#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=lstm
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --gres=gpu:1

cd /gpfs/home/kgz2437/pathology_parsing/testing/NLU/Natural-Language-Understanding/hw2-main
python lstm_job.py False 0.01 16
python lstm_job.py False 0.05 16
python lstm_job.py False 0.005 16
