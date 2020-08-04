#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:10:00
#SBATCH -p standard
#SBATCH -A bhdvcs
#SBATCH --job-name=mw6es_extraction_net

#SBATCH --output=mw6es_extraction_net%A_%a.out
#SBATCH --error=mw6es_extraction_net_%A_%a.error

python ANN3.py ${SLURM_ARRAY_TASK_ID}

#sbatch --array=0-13 myslurm_array.sh