#!/bin/bash
# Compute LPSA matrix A = X@phi using parallel workers (calculating superframes in X implicitely)
#SBATCH --ntasks-per-core=1

module purge
module load anaconda
conda activate myenv_nlsa

WORKER_ID=$SLURM_ARRAY_TASK_ID   # From 0 to ...

# pass sbatch arguments to python
python -m dynamics_retrieval.calculate_aj $WORKER_ID "$@"