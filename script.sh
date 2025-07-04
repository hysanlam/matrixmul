#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=math-454
#SBATCH --account=math-454

module purge
module load gcc cuda

srun nvprof ./matrixmul
