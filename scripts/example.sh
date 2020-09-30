#!/bin/bash

#SBATCH --job-name=zonghaiyao/run_boxmodel/uhsv1n4l
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=slurm_output/slurm_output.out
#SBATCH --error=slurm_output/slurm_output.error
#SBATCH --array=1-10

wandb agent zonghaiyao/run_boxmodel/uhsv1n4l