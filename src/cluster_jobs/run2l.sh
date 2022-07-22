#!/bin/bash
echo "run.sh works"
source /conda/etc/profile.d/conda.sh
conda activate bachelor_env
python --version
python cluster_job2l.py $1 $2 $3 $4 $5 $6 $7
