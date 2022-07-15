#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --constraint="GPUx1&V100"
#SBATCH --chdir /beegfs/desy/user/hilbertm/slurm_out
#SBATCH --job-name  testi
#SBATCH --output    testi-%j.out
#SBATCH --error     testi-%j.err

unset LD_PRELOAD
source /etc/profile.d/modules.sh
cd /beegfs/desy/user/hilbertm
singularity exec --nv --bind /beegfs/desy/user/hilbertm /beegfs/desy/user/hilbertm/singularity_container.sif /beegfs/desy/user/hilbertm/run.sh 0.0 25 50 75 12 8 4 256
