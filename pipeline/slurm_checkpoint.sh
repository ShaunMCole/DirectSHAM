#!/bin/bash
#SBATCH --job-name=checkpoint
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0                  # request all memory on the node
#SBATCH --time=02:00:00
#SBATCH --output=logs/checkpoint_%x_%j.out
#SBATCH --error=logs/checkpoint_%x_%j.err

# slurm_checkpoint.sh — Submit make_checkpoint.py for one region and one
# set of scatter parameters.
#
# Do not run this script directly with sbatch — use run_pipeline.sh which
# sets the required environment variables before submitting:
#
#   REG          : N or S
#   OBS_SEED     : observer position seed
#   SIG_LGV      : Vpeak scatter in dex
#   MIN_VPEAK    : minimum Vpeak threshold [km/s]  (optional, has default)
#   SNAP_INDEX   : Uchuu snapshot index             (optional, has default)
#
# Example (from run_pipeline.sh):
#   export REG=N OBS_SEED=0 SIG_LGV=0.10
#   sbatch slurm_checkpoint.sh

set -euo pipefail

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started:    $(date)"
echo "REG:        ${REG}"
echo "OBS_SEED:   ${OBS_SEED}"
echo "SIG_LGV:    ${SIG_LGV}"
echo "MIN_VPEAK:  ${MIN_VPEAK:-70.0}"
echo "SNAP_INDEX: ${SNAP_INDEX:-45}"
echo "=========================================="

# Load the DESI software environment.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$PYTHONPATH

# Move into the pipeline directory so relative imports work.
cd "$(dirname "$0")"

# Ensure the logs directory exists.
mkdir -p logs

python make_checkpoint.py \
    --reg        "${REG}" \
    --obs_seed   "${OBS_SEED}" \
    --sig_lgv    "${SIG_LGV}" \
    --min_vpeak  "${MIN_VPEAK:-70.0}" \
    --n_neigh_vpeak "${N_NEIGH_VPEAK:-1000}" \
    --snapshot_index "${SNAP_INDEX:-45}"

echo "=========================================="
echo "Finished:   $(date)"
echo "=========================================="
