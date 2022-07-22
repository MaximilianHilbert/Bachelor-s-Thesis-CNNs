import os
import subprocess
import numpy as np
#HP 2 layer auto num_units 
HP_LR=[0.001, 0.005, 0.01, 0.05]
HP_NOISE_LEVEL=[0.15]
HP_NUM_FILTER_1=np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
HP_NUM_FILTER_2=HP_NUM_FILTER_1*2
HP_KERNEL_SIZE_1=np.array([4, 8, 10, 14, 18, 22])
HP_KERNEL_SIZE_2=HP_KERNEL_SIZE_1/2
HP_BATCH_SIZE=[1024, 512, 256, 128]

# HP_LR = [0.001]
# HP_NOISE_LEVEL = [np.round(x, 2) for x in np.arange(0, 1.0, 1.0/20)]
# HP_NUM_FILTER_1 = [100]
# HP_NUM_FILTER_2 = [50]
# HP_KERNEL_SIZE_1 = [12]
# HP_KERNEL_SIZE_2 = [8]
# HP_BATCH_SIZE = [512]


param_lst = []
for noise_level in HP_NOISE_LEVEL:
    for num_filter_1, num_filter_2 in zip(HP_NUM_FILTER_1, HP_NUM_FILTER_2):
        for kernel_size_1, kernel_size_2 in zip(HP_KERNEL_SIZE_1, HP_KERNEL_SIZE_2):
            for lr in HP_LR:
                for batch_size in HP_BATCH_SIZE:
                    hparams = [lr, noise_level, num_filter_1,
                                num_filter_2, kernel_size_1, kernel_size_2, batch_size]
                    param_lst.append(hparams)

for idx, list_el in enumerate(param_lst):
    learning_rate = float(list_el[0])
    noise_level = float(list_el[1])
    num_filter_1 = float(list_el[2])
    num_filter_2 = float(list_el[3])
    kernel_size_1 = float(list_el[4])
    kernel_size_2 = float(list_el[5])
    batch_size = float(list_el[6])
    String = f'#!/bin/bash\n#SBATCH --partition=allgpu\n#SBATCH --time=0-03:00:00\n#SBATCH --nodes=1\n#SBATCH --constraint="GPUx1&V100"\n#SBATCH --chdir /beegfs/desy/user/hilbertm/slurm_out\n#SBATCH --job-name  2lHP\n#SBATCH --output    2lHP-%j.out\n#SBATCH --error     2lHP-%j.err\nunset LD_PRELOAD\nsource /etc/profile.d/modules.sh\ncd /beegfs/desy/user/hilbertm\nsingularity exec --nv --bind /beegfs/desy/user/hilbertm /beegfs/desy/user/hilbertm/singularity_container.sif /beegfs/desy/user/hilbertm/run2l.sh {learning_rate} {noise_level} {num_filter_1} {num_filter_2} {kernel_size_1} {kernel_size_2} {batch_size}'
    if not os.path.exists("jobs/"):
        os.makedirs("jobs/")
    with open(f"jobs/job_{idx}", "w") as file:
        file.write(String)
    subprocess.call(f"chmod u+x jobs/job_{idx}", shell=True)
    subprocess.call(f"sbatch jobs/job_{idx}", shell=True)
