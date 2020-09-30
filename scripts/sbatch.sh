#!/bin/bash

#SBATCH --job-name=c0krlu27-%j
#SBATCH --gres=gpu:1
#SBATCH --partition=1080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=/mnt/nfs/work1/llcao/zonghaiyao/ontology_alignment/box_for_ontology_alignment/log/%j.out
#SBATCH --error=/mnt/nfs/work1/llcao/zonghaiyao/ontology_alignment/box_for_ontology_alignment/log/%j.error
#SBATCH --array=1-10

wandb agent iesl-boxes/box_for_ontology_alignment/c0krlu27