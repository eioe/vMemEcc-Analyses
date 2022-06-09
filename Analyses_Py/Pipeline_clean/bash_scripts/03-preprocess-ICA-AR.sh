#!/bin/bash -l 
# specify the indexes of the job array elements
#SBATCH --array=0-18
# Standard output and error: 
#SBATCH -o ./job.out.%j        # Standard output, %A = job ID, %a = job array index 
#SBATCH -e ./job.err.%j        # Standard error, %A = job ID, %a = job array index 
# Initial working directory: 
#SBATCH -D /ptmp/fklotzsche/Experiments/vMemEcc/
# Job Name: 
#SBATCH -J hansi_ftw
# Queue (Partition): 
#SBATCH --partition=general 
# Number of nodes and MPI tasks per node: 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=15000MB
# 
#SBATCH --mail-type=all 
#SBATCH --mail-user=klotzsche@cbs.mpg.de 
# 
#SBATCH --time=00:20:00
 
module load anaconda/3
module load mkl
conda activate mne
 
# Run the program: 
srun python3.9 ./vme_analysis/Analyses_Py/03_preprocess-ica-ar.py $SLURM_ARRAY_TASK_ID
