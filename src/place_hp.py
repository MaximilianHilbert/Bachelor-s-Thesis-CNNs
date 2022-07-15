import os
import subprocess
import numpy as np

# HP_NOISE_LEVEL=[0.0]  #only one
# HP_NUM_FILTER_1=[5, 10, 15, 20, 25]
# HP_NUM_FILTER_2=[10, 20, 30, 40, 50]
# HP_NUM_FILTER_3=[20, 40, 60, 80, 100]

# HP_KERNEL_SIZE_1=[10, 12, 14, 16, 18, 20, 22, 24]
# HP_KERNEL_SIZE_2=[8, 10, 12, 14, 16, 18, 20, 22]
# HP_KERNEL_SIZE_3=[6, 8, 10, 12, 14, 16, 18, 20]
# HP_BATCH_SIZE=[1024, 512, 128]

HP_NOISE_LEVEL = [np.round(x, 2) for x in np.arange(0, 1.0, 1.0/20)]
HP_NUM_FILTER_1 = [25]
HP_NUM_FILTER_2 = [50]
HP_NUM_FILTER_3 = [100]
HP_KERNEL_SIZE_1 = [10]
HP_KERNEL_SIZE_2 = [8]
HP_KERNEL_SIZE_3 = [6]
HP_BATCH_SIZE = [512]


param_lst = []
for noise_level in HP_NOISE_LEVEL:
    for num_filter_1, num_filter_2, num_filter_3 in zip(HP_NUM_FILTER_1, HP_NUM_FILTER_2, HP_NUM_FILTER_3):
        for kernel_size_1, kernel_size_2, kernel_size_3 in zip(HP_KERNEL_SIZE_1, HP_KERNEL_SIZE_2, HP_KERNEL_SIZE_3):
            for batch_size in HP_BATCH_SIZE:
                hparams = [noise_level, num_filter_1,
                            num_filter_2, num_filter_3, kernel_size_1, kernel_size_2, kernel_size_3, batch_size]
                param_lst.append(hparams)
for i in range(5):
    for idx, list_el in enumerate(param_lst):
        noise_level = float(list_el[0])
        num_filter_1 = float(list_el[1])
        num_filter_2 = float(list_el[2])
        num_filter_3 = float(list_el[3])
        kernel_size_1 = float(list_el[4])
        kernel_size_2 = float(list_el[5])
        kernel_size_3 = float(list_el[6])
        batch_size = float(list_el[7])

        String = f'#!/bin/bash\n#SBATCH --partition=allgpu\n#SBATCH --time=0-00:30:00\n#SBATCH --nodes=1\n#SBATCH --constraint="GPUx1&V100"\n#SBATCH --chdir /beegfs/desy/user/hilbertm/slurm_out\n#SBATCH --job-name  HPNorm\n#SBATCH --output    HPNorm-%j.out\n#SBATCH --error     HPNorm-%j.err\nunset LD_PRELOAD\nsource /etc/profile.d/modules.sh\ncd /beegfs/desy/user/hilbertm\nsingularity exec --nv --bind /beegfs/desy/user/hilbertm /beegfs/desy/user/hilbertm/singularity_container.sif /beegfs/desy/user/hilbertm/run.sh {noise_level} {num_filter_1} {num_filter_2} {num_filter_3} {kernel_size_1} {kernel_size_2} {kernel_size_3} {batch_size}'
        if not os.path.exists("jobs/"):
            os.makedirs("jobs/")
        with open(f"jobs/job_{idx}", "w") as file:
            file.write(String)
        subprocess.call(f"chmod u+x jobs/job_{idx}", shell=True)
        subprocess.call(f"sbatch jobs/job_{idx}", shell=True)
