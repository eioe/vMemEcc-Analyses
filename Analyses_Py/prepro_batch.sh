#!/bin/bash -l 
          # specify the indexes of the job array elements 
# Standard output and error: 
#SBATCH -o job_%A_%a.out        # Standard output, %A = job ID, %a = job array index 
#SBATCH -e job_%A_%a.err        # Standard error, %A = job ID, %a = job array index 
# Initial working directory: 
#SBATCH -D ./ptmp/fklotzsche
# Job Name: 
#SBATCH -J prepro_test 
# Queue (Partition): 
#SBATCH --partition=general 
# Number of nodes and MPI tasks per node: 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=40 
# 
#SBATCH --mail-type=none 
#SBATCH --mail-user=klotzsche@cbs.mpg.de 
# 
# Wall clock limit: 
 
 
# Run the program: 
srun ./Experiments/vMemEcc/vme_analysis/Analyses_Py/02.1-preprocess_data_batch.py 