#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/scratch/kgz2437/bert_finetuning/slurm_%j.out
#SBATCH --error=/scratch/kgz2437/bert_finetuning/slurm_%j.err
#SBATCH --account=ds_ga_1012_2023sp
#SBATCH --partition=nls8-v100-1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --requeue

singularity exec --nv --bind /scratch/kgz2437/ --overlay /scratch/kgz2437/overlay-25GB-500k.ext3:rw /scratch/kgz2437/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "source /ext3/env.sh; conda activate; cd /scratch/kgz2437/Natural-Language-Understanding/hw3-main; python train_model.py"

