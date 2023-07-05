#!/bin/bash
#SBATCH --array=0-9
#SBATCH --partition=day        
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00            

WORKER_ID=$SLURM_ARRAY_TASK_ID   # From 0 to 15

module load matlab/2015b

matlab -nodisplay -nosplash -nodesktop -r "f_grab_scalable_hklI_asu_uid($WORKER_ID)"

