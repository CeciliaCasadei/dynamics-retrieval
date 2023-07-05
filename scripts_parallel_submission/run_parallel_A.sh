#!/bin/bash

#SBATCH --ntasks-per-core=1

module purge
module load anaconda
conda activate dynamics-retrieval

WORKER_ID=$SLURM_ARRAY_TASK_ID   # From 0 to ...

# pass sbatch arguments to python
python -m dynamics_retrieval.calculate_A_parallel $WORKER_ID "$@"