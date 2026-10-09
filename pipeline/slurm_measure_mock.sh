#!/bin/bash
#SBATCH --job-name=xi_mock
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/global/homes/s/smcole/DirectSHAM/logs/xi_mock_%x_%j.out
#SBATCH --error=/global/homes/s/smcole/DirectSHAM/logs/xi_mock_%x_%j.err

# slurm_measure_mock.sh — Submit measure_mock_clustering.py for one mock
# realisation and one set of sample selection parameters.
#
# Environment variables (set by run_pipeline.sh or manually):
#
#   OBS_SEED : observer position seed
#   SIG_LGV  : Vpeak scatter in dex
#   MR_CUT   : absolute magnitude cut
#   ZMIN     : minimum redshift
#   ZMAX     : maximum redshift
#   CAP      : NGC, SGC, or NGC+SGC (default: NGC+SGC)
#   NTHREADS : number of pycorr threads (default: 64)

set -euo pipefail

echo "=========================================="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "OBS_SEED: ${OBS_SEED}"
echo "SIG_LGV:  ${SIG_LGV}"
echo "MR_CUT:   ${MR_CUT}"
echo "ZMIN:     ${ZMIN}"
echo "ZMAX:     ${ZMAX}"
echo "CAP:      ${CAP:-NGC+SGC}"
echo "=========================================="

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$PYTHONPATH

cd "$(dirname "$0")"
mkdir -p logs

python measure_mock_clustering.py \
    --obs_seed  "${OBS_SEED}" \
    --sig_lgv   "${SIG_LGV}" \
    --mr_cut    "${MR_CUT}" \
    --zmin      "${ZMIN}" \
    --zmax      "${ZMAX}" \
    --cap       "${CAP:-NGC+SGC}" \
    --nthreads  "${NTHREADS:-64}"

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
