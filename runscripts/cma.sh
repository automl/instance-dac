#!/bin/bash

#SBATCH --error=cmaes.err
#SBATCH --job-name=instance_dac_cmaes
#SBATCH --mem=100GB
#SBATCH --output=cmaes.out
#SBATCH --partition=ai,tnt
#SBATCH --time=200
#SBATCH --array=1-21

source /bigwork/nhwpmoha/miniconda3/etc/profile.d/conda.sh
conda activate instance_dac
module load cuDNN/8.8.0.121-CUDA-12.0.0
python instance_dac/train.py +benchmark=cmaes +inst/cmaes=default seed=${SLURM_ARRAY_TASK_ID} -m