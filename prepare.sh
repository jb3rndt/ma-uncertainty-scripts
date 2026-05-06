#!/bin/bash -eux
#SBATCH --job-name=prepare
#SBATCH --account=sci-naumann
#SBATCH --partition=gpu-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --output=logs/prepare_%j.log

eval "$(conda shell.bash hook)"
conda activate ma

~/ollama/bin/ollama serve &
sleep 1
python prepare_dismis.py

kill $!
