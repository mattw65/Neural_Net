#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH -p standard
#SBATCH -A spinquest
#SBATCH --job-name=replica.py

#SBATCH --output=replica_results.out
#SBATCH --error=replica_results.error

python replica.py ${SLURM_ARRAY_TASK_ID}