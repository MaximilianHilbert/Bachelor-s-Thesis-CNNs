#!/bin/bash
#SBATCH --partition=all
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --chdir /beegfs/desy/user/hilbertm/slurm_out
#SBATCH --job-name  testi
#SBATCH --output    testi-%j.out
#SBATCH --error     testi-%j.err
unset LD_PRELOAD
source /etc/profile.d/modules.sh
cd /beegfs/desy/user/hilbertm
singularity exec --nv --bind /beegfs/desy/user/hilbertm /beegfs/desy/user/hilbertm/singularity_container.sif /beegfs/desy/user/hilbertm/run_one_time.sh
