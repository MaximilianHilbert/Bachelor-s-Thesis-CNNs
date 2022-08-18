#!/bin/bash
echo "run.sh works"
source /conda/etc/profile.d/conda.sh
conda activate bachelor_env
python --version
python mlp_cluster_job.py
