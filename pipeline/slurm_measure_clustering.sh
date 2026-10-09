#!/bin/bash
#SBATCH --job-name=xi_data
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/xi_data_%x_%j.out
#SBATCH --error=logs/xi_data_%x_%j.err

# slurm_measure_clustering.sh — Submit measure_clustering.py for one set of
# sample selection parameters.
#
# Environment variables (set by run_pipeline.sh or manually):
#
#   MR_CUT  : absolute magnitude cut (e.g. -21.5)
#   ZMIN    : minimum redshift       (e.g. 0.02)
#   ZMAX    : maximum redshift       (e.g. 0.42)
#   NJACK   : jackknife regions      (0 = none)

set -euo pipefail

echo "=========================================="
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "MR_CUT:  ${MR_CUT}"
echo "ZMIN:    ${ZMIN}"
echo "ZMAX:    ${ZMAX}"
echo "NJACK:   ${NJACK:-0}"
echo "=========================================="

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$PYTHONPATH

cd "$(dirname "$0")"
mkdir -p logs

python measure_clustering.py \
    --mr_cut "${MR_CUT}" \
    --zmin   "${ZMIN}" \
    --zmax   "${ZMAX}" \
    --njack  "${NJACK:-0}"

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
