#!/bin/bash -eux
#SBATCH --job-name=assess
#SBATCH --account=sci-naumann
#SBATCH --partition=cpu-batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/assess_%j.log

eval "$(conda shell.bash hook)"
conda activate ma

export FAISS_DISABLE_CPU_FEATURES="AVX2,AVX512"
python assess.py
