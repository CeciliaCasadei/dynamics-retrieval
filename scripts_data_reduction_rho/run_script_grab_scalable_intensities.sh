#!/bin/bash
#SBATCH --array=0-10
#SBATCH --partition=hour          # Using 'hourly' will grant higher priority

##SBATCH --ntasks-per-core=1        # Force no Hyper-Threading, will run 1 task per core
##SBATCH --mem=110000

##SBATCH --mem=352000               # ???? We want to use the whole memory
##SBATCH --mem=256000               # ???? We want to use the whole memory
##SBATCH --time=00:30:00            # Define max time job will run
##SBATCH --output=myscript.out      # Define your output file
##SBATCH --error=myscript.err       # Define your error file

##WORKER_ID=${SLURM_ARRAY_JOB_ID:-1}


WORKER_ID=$SLURM_ARRAY_TASK_ID      # From 0 to 5

module load matlab/2015b

matlab -nodisplay -nosplash -nodesktop -r "script_grab_scalable_intensities('alldark', $WORKER_ID)"