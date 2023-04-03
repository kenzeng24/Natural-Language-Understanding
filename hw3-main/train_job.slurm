#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/scratch/kgz2437/bert_finetuning/bert_tiny.out
#SBATCH --error=/scratch/kgz2437/bert_finetuning/bert_tiny.err
#SBATCH --account=ds_ga_1012_2023sp
#SBATCH --partition=nls8-v100-1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --requeue

singularity exec --nv\
  --overlay /scratch/kgz2437/overlap-25GB-500k.ext3:ro \
  /scratch/kgz2437/nlu_hw3/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source / ext3/env.sh ; conda activate base; cd /scratch/kgz2437/Natural-Language-Processing/hw3-main ; python train_model.py True; python train_model.py False"

