#!/bin/bash -eux
#SBATCH --job-name=assess
#SBATCH --account=sci-naumann
#SBATCH --partition=cpu-batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/assess_%j.log
#SBATCH -C "CPU_SKU:5220S"

eval "$(conda shell.bash hook)"
conda activate ma

echo "===== Benchmark machine info ====="
date
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-N/A}"
echo "Node list: ${SLURM_JOB_NODELIST:-N/A}"
echo

echo "----- Slurm allocation -----"
echo "Tasks: ${SLURM_NTASKS:-N/A}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "CPUs on node (allocated context): ${SLURM_CPUS_ON_NODE:-N/A}"
echo "Memory per node: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Memory per CPU: ${SLURM_MEM_PER_CPU:-N/A}"
echo "GRES: ${SLURM_JOB_GRES:-N/A}"
echo

echo "----- CPU hardware -----"
lscpu | egrep 'Architecture|Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)|NUMA node\(s\)'
echo

echo "----- Memory hardware -----"
# total physical RAM in GiB
awk '/MemTotal/ {printf "MemTotal: %.2f GiB\n", $2/1024/1024}' /proc/meminfo
echo

echo "----- Slurm node record -----"
# Shows configured node cores/memory/features from Slurm
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  scontrol show node "$SLURM_JOB_NODELIST" | egrep 'NodeName=|Arch=|CoresPerSocket=|CPUAlloc=|CPUTot=|RealMemory=|AllocMem=|ThreadsPerCore=|Sockets='
fi

echo "===== End machine info ====="

export FAISS_DISABLE_CPU_FEATURES="AVX2,AVX512"
python assess.py
