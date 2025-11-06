#!/bin/bash
#SBATCH --job-name=car-robustness
#SBATCH --array=1-25
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# module load python/3.10
# module load cuda/12.1
# source ~/venvs/vehicle-env/bin/activate

# Read parameters for this job
ETA=$(awk "NR==${SLURM_ARRAY_TASK_ID} {print \$1}" strongrobust_params.txt)
EPS=$(awk "NR==${SLURM_ARRAY_TASK_ID} {print \$2}" strongrobust_params.txt)

echo "Running verification with eta=${ETA}, epsilon=${EPS}"

# Create output directory
mkdir -p results

# Run the verification command
vehicle verify \
  --specification car-robust-safety.vcl \
  --network nnModel:nn_model.onnx \
  --parameter epsilon:${EPS} \
  --parameter eta:${ETA} \
  --dataset trainingInputs:trainingDatas.idx \
  --dataset trainingLabels:trainingLabels.idx \
  --verifier Marabou \
  --property strongRobust \
  > results/eta_${ETA}_eps_${EPS}.log 2>&1
