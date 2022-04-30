#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 00:60:00
#SBATCH --gres=gpu:1

##change this before start the job
export fc_nodes=12
singularity exec /ocean/containers/ngc/pytorch/pytorch_latest.sif bash -c "cd ~/294-082-final-project && python -m model_baseline.resnet-eurosat --fc_nodes $fc_nodes"
