import os
import subprocess
import numpy as np
# HP_LR=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# HP_NUM_UNITS=[50, 70, 90, 100]
# HP_NOISE_LEVEL=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# HP_NUM_FILTER_1=[30, 50, 70, 90, 100]
# HP_NUM_FILTER_2=[15, 25, 35, 45, 50]
# HP_KERNEL_SIZE_1=[12, 20, 28, 36, 44]
# HP_KERNEL_SIZE_2=[8, 10, 14, 18, 22]
# HP_BATCH_SIZE=[1024, 512, 128]
HP_LR = [0.001]
HP_NUM_UNITS = [100]
HP_NOISE_LEVEL = [np.round(x, 2) for x in np.arange(0, 1.0, 1.0/20)]
HP_NUM_FILTER_1 = [100]
HP_NUM_FILTER_2 = [50]
HP_KERNEL_SIZE_1 = [12]
HP_KERNEL_SIZE_2 = [8]
HP_BATCH_SIZE = [512]


param_lst = []
for num_units in HP_NUM_UNITS:
    for noise_level in HP_NOISE_LEVEL:
        for num_filter_1, num_filter_2 in zip(HP_NUM_FILTER_1, HP_NUM_FILTER_2):
            for kernel_size_1, kernel_size_2 in zip(HP_KERNEL_SIZE_1, HP_KERNEL_SIZE_2):
                for lr in HP_LR:
                    for batch_size in HP_BATCH_SIZE:
                        hparams = [lr, noise_level, num_units, num_filter_1,
                                   num_filter_2, kernel_size_1, kernel_size_2, batch_size]
                        param_lst.append(hparams)
for i in range(5):
    for idx, list_el in enumerate(param_lst):
        learning_rate = float(list_el[0])
        noise_level = float(list_el[1])
        num_units = float(list_el[2])
        num_filter_1 = float(list_el[3])
        num_filter_2 = float(list_el[4])
        kernel_size_1 = float(list_el[5])
        kernel_size_2 = float(list_el[6])
        batch_size = float(list_el[7])
        String = f'#!/bin/bash\n#SBATCH --partition=allgpu\n#SBATCH --time=0-00:30:00\n#SBATCH --nodes=1\n#SBATCH --constraint="GPUx1&V100"\n#SBATCH --chdir /beegfs/desy/user/hilbertm/slurm_out\n#SBATCH --job-name  HPNorm\n#SBATCH --output    HPNorm-%j.out\n#SBATCH --error     HPNorm-%j.err\nunset LD_PRELOAD\nsource /etc/profile.d/modules.sh\ncd /beegfs/desy/user/hilbertm\nsingularity exec --nv --bind /beegfs/desy/user/hilbertm /beegfs/desy/user/hilbertm/singularity_container.sif /beegfs/desy/user/hilbertm/run.sh {learning_rate} {noise_level} {num_units} {num_filter_1} {num_filter_2} {kernel_size_1} {kernel_size_2} {batch_size}'
        if not os.path.exists("jobs/"):
            os.makedirs("jobs/")
        with open(f"jobs/job_{idx}", "w") as file:
            file.write(String)
        subprocess.call(f"chmod u+x jobs/job_{idx}", shell=True)
        subprocess.call(f"sbatch jobs/job_{idx}", shell=True)
