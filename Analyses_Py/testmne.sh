#!/bin/bash -l 
# specify the indexes of the job array elements
#SBATCH --array=0-2 
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
#SBATCH --cpus-per-task=1 
#SBATCH --mem=2000

#SBATCH --mail-type=all 
#SBATCH --mail-user=klotzsche@cbs.mpg.de 
# 
#SBATCH --time=00:01:00
 
module load anaconda/3/2021.05
conda activate mne
conda list
conda install -c conda-forge mne-base h5io h5py pymatreader

# Run the program: 
srun python ./vme_analysis/Analyses_Py/testmne.py $SLURM_ARRAY_TASK_ID
