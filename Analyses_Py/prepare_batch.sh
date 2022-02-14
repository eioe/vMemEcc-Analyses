#!/bin/bash -l 
# specify the indexes of the job array elements
#SBATCH --array=0-26 
# Standard output and error: 
#SBATCH -o ./job.out.%j        # Standard output, %A = job ID, %a = job array index 
#SBATCH -e ./job.err.%j        # Standard error, %A = job ID, %a = job array index 
# Initial working directory: 
#SBATCH -D /ptmp/fklotzsche/Experiments/vMemEcc/
# Job Name: 
#SBATCH -J prepro_test 
# Queue (Partition): 
#SBATCH --partition=general 
# Number of nodes and MPI tasks per node: 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 
#SBATCH --mem=20000
# 
#SBATCH --mail-type=all 
#SBATCH --mail-user=klotzsche@cbs.mpg.de 
# 
#SBATCH --time=00:05:00
 
module load anaconda/3
conda activate mne
 
# Run the program: 
srun python ./vme_analysis/Analyses_Py/01.1-epoch_data.py $SLURM_ARRAY_TASK_ID            #02.1_preprocess_data_batch.py $SLURM_ARRAY_TASK_ID
