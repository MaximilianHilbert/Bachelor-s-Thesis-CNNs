import os
import subprocess
import numpy as np
from tqdm import tqdm
#HP3 layer auto num_units 
HP_NOISE_LEVEL=[0.3]

HP_NUM_FILTER_1=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
HP_NUM_FILTER_2=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
HP_NUM_FILTER_3=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

HP_KERNEL_SIZE_1 = [4, 5, 6, 7, 8, 9, 10]
HP_KERNEL_SIZE_2 = [3, 4, 5, 6, 7, 8, 9]
HP_KERNEL_SIZE_3 = [2, 3, 4, 5, 6, 7, 8]
HP_BATCH_SIZE=[512, 256]


# # HP_NOISE_LEVEL = [np.round(x, 2) for x in np.arange(0, 1.0, 1.0/20)]

# # HP_NOISE_LEVEL = [0.3]
# # HP_NUM_FILTER_1 = [25]
# # HP_NUM_FILTER_2 = [50]
# # HP_NUM_FILTER_3=[75]
# # HP_KERNEL_SIZE_1 = [12]
# # HP_KERNEL_SIZE_2 = [8]
# # HP_KERNEL_SIZE_3 = [4]
# # HP_BATCH_SIZE = [256]

param_lst = []
for noise_level in HP_NOISE_LEVEL:
    for num_filter_1, num_filter_2, num_filter_3 in zip(HP_NUM_FILTER_1, HP_NUM_FILTER_2, HP_NUM_FILTER_3):
        for kernel_size_1 in HP_KERNEL_SIZE_1:
            for kernel_size_2 in HP_KERNEL_SIZE_2:
                for kernel_size_3 in HP_KERNEL_SIZE_3:
                    for batch_size in HP_BATCH_SIZE:
                        hparams = [noise_level, num_filter_1,
                                    num_filter_2, num_filter_3, kernel_size_1, kernel_size_2, kernel_size_3, batch_size]
                        param_lst.append(hparams)

for idx in tqdm(range(len(param_lst))):
    list_el=param_lst[idx]
    noise_level = float(list_el[0])
    num_filter_1 = float(list_el[1])
    num_filter_2 = float(list_el[2])
    num_filter_3 = float(list_el[3])
    kernel_size_1 = float(list_el[4])
    kernel_size_2 = float(list_el[5])
    kernel_size_3 = float(list_el[6])
    batch_size = float(list_el[7])
    
    String = f'#!/bin/bash\n#SBATCH --partition=allgpu\n#SBATCH --time=0-00:30:00\n#SBATCH --nodes=1\n#SBATCH --constraint="GPUx1&V100"\n#SBATCH --chdir /beegfs/desy/user/hilbertm/slurm_out\n#SBATCH --job-name  3lHP\n#SBATCH --output    3lHP-%j.out\n#SBATCH --error     3lHP-%j.err\nunset LD_PRELOAD\nsource /etc/profile.d/modules.sh\ncd /beegfs/desy/user/hilbertm\nsingularity exec --nv --bind /beegfs/desy/user/hilbertm /beegfs/desy/user/hilbertm/singularity_container.sif /beegfs/desy/user/hilbertm/run3l.sh {noise_level} {num_filter_1} {num_filter_2} {num_filter_3} {kernel_size_1} {kernel_size_2} {kernel_size_3} {batch_size}'
    if not os.path.exists("jobs/"):
        os.makedirs("jobs/")
    with open(f"jobs/job_{idx}", "w") as file:
        file.write(String)
    subprocess.call(f"chmod u+x jobs/job_{idx}", shell=True)
    subprocess.call(f"sbatch jobs/job_{idx}", shell=True)
