#!/bin/bash
#SBATCH --job-name=car_control_training
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=PGR-Standard

module purge
module load anaconda3

# Initialize conda and activate env
source $HOME/.bashrc
conda activate car_control_env

# Debug env info
echo "=== Environment Information ==="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "================================="

python simpler_training.py

echo "Training completed!"