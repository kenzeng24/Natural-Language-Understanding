#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/scratch/${USER}/%j_%x.out
#SBATCH --error=/scratch/${USER}/%j_%x.err
#SBATCH --account=ds_ga_1012_2023sp
#SBATCH --partition=nls8-v100-1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --requeue


# Problem 2a: Write a script to be executed by your HPC job in order to fine-tune a BERT-tiny model.