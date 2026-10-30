#!/bin/bash
# slurm_lightcone_mock.sh
# -----------------------
# Run make_mock_snap.py for all 6 snapshots x 2 regions = 12 jobs.
# The matching step is fast (~minutes per job on a login node, but
# running via SLURM avoids timeout risk for the lower-z snapshots
# which have large random catalogues).
#
# Uses a 12-element job array: tasks 0-5 = region N, tasks 6-11 = region S.
#
# Submit AFTER all checkpoint jobs have completed:
#   sbatch --dependency=afterok:<checkpoint_jobid> slurm_lightcone_mock.sh
#
# Or submit independently once checkpoints are done:
#   sbatch slurm_lightcone_mock.sh

#SBATCH --job-name=lc_mock
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/lc_mock_%A_%a.out
#SBATCH --error=logs/lc_mock_%A_%a.err
#SBATCH --array=0-11

# ── Model parameters ─────────────────────────────────────────────────────────
SIG_LGV_REF=0.18
SIG_LGV_SLOPE=0.30
F_COLOUR=0.00
ALPHA=0.00

# ── Map array task ID → (snapshot, region) ───────────────────────────────────
# Tasks 0-5: region N,  tasks 6-11: region S
SNAPSHOTS=(50 47 45 43 41 40 50 47 45 43 41 40)
REGIONS=(   N  N  N  N  N  N  S  S  S  S  S  S)

SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}
REG=${REGIONS[$SLURM_ARRAY_TASK_ID]}

# ── Environment ───────────────────────────────────────────────────────────────
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/ShaunColeDESI:$HOME/DirectSHAM/pipeline:$PYTHONPATH

mkdir -p logs

echo "Starting mock for snapshot ${SNAP} region ${REG}  (array task ${SLURM_ARRAY_TASK_ID})"
date

python $HOME/DirectSHAM/pipeline/make_mock_snap.py \
    --snapshot     ${SNAP}          \
    --reg          ${REG}           \
    --sig_lgv_ref  ${SIG_LGV_REF}  \
    --sig_lgv_slope ${SIG_LGV_SLOPE} \
    --f_colour     ${F_COLOUR}      \
    --alpha        ${ALPHA}

echo "Finished snapshot ${SNAP} region ${REG}"
date
