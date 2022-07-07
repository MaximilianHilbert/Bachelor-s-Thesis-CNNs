#!/bin/bash
echo "run_one_time.sh works"
source /conda/etc/profile.d/conda.sh
conda activate bachelor_env
python --version
python lib/data_gen.py
