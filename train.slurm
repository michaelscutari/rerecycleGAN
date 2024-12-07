#!/bin/bash
#SBATCH --job-name=rerecycleGAN
#SBATCH --gres=gpu:a5000:4  # Request 4 A100 GPUs
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8  # Allocate 8 CPU threads
#SBATCH --time=48:00:00  # Adjust time limit as needed
#SBATCH --output=output_%j.log  # Save output logs
#SBATCH --error=error_%j.err
\
module load cuda/cuda-12.1  # Load necessary modules if required
source ~/.bashrc  # Activate your conda environment

eval "$(micromamba shell hook --shell bash)"

micromamba activate ece661

export WANDB_API_KEY=2cffe09f8f0eb9f83180bfc498a98e1476803894

python train.py