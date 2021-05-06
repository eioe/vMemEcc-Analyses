#!/bin/bash -l 
# specify the indexes of the job array elements
# Standard output and error: 
#SBATCH -o ./job.out.%j        # Standard output, %A = job ID, %a = job array index 
#SBATCH -e ./job.err.%j        # Standard error, %A = job ID, %a = job array index 
# Initial working directory: 
#SBATCH -D /ptmp/fklotzsche/Experiments/vMemEcc/
# Job Name: 
#SBATCH -J hansi_ftw
# Queue (Partition): 
#SBATCH --partition=gpu
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:2
# #SBATCH --mem=61000
# Number of nodes and MPI tasks per node: 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
# 
#SBATCH --mail-type=all 
#SBATCH --mail-user=klotzsche@cbs.mpg.de 
# 
#SBATCH --time=12:00:00
 
module load anaconda/3/2019.03
module load cuda/10.0
module load tensorflow/gpu/1.14.0
conda activate DLC-GPU-custom
conda env list
module list

# Run the program: 
srun python ./vme_analysis/Analyses_Py/EyeTracking/Eye_Tracking_Analysis.py 
