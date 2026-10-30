#!/bin/bash
# slurm_lightcone_checkpoint.sh
# ------------------------------
# Run make_checkpoint_snap.py for all 6 lightcone snapshots.
# Each job reads the full Uchuu halo catalogue for one snapshot and
# writes two checkpoint files (N and S) to the lightcone model directory.
#
# Resource requirements: same as slurm_checkpoint.sh — exclusive node,
# ~1 hr per snapshot.  Run as a job array so all 6 can run in parallel
# (subject to queue availability).
#
# Submit with:
#   sbatch slurm_lightcone_checkpoint.sh
#
# NOTE: snap45 output can be symlinked from the existing cut-sky checkpoint
# rather than recomputed — see comment below.

#SBATCH --job-name=lc_checkpoint
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/lc_checkpoint_%A_%a.out
#SBATCH --error=logs/lc_checkpoint_%A_%a.err
#SBATCH --array=0-5

# ── Model parameters ─────────────────────────────────────────────────────────
SIG_LGV_REF=0.18
SIG_LGV_SLOPE=0.30
F_COLOUR=0.00
ALPHA=0.00

# ── Snapshot list (must match LIGHTCONE_SHELLS in config.py) ─────────────────
SNAPSHOTS=(50 47 45 43 41 40)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

# ── NOTE on snap45 ───────────────────────────────────────────────────────────
# snap45 (z=0.19) was already processed by the cut-sky pipeline.
# You can skip this job for SLURM_ARRAY_TASK_ID=2 and instead symlink:
#
#   cd $DATA_DATE_DIR/lightcone_sig_lgv0.18_0.30_fc0.00_al0.00/
#   ln -s ../../seed000_sig_lgv0.18_0.30_fc0.00_al0.00/subhalos_N_70.0_...fits \
#         subhalos_N_snap45.fits
#   ln -s ../../seed000_sig_lgv0.18_0.30_fc0.00_al0.00/subhalos_S_70.0_...fits \
#         subhalos_S_snap45.fits
#
# If you do this, cancel array task 2 before submitting, or add:
# if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then exit 0; fi

# ── Environment ───────────────────────────────────────────────────────────────
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/ShaunColeDESI:$HOME/DirectSHAM/pipeline:$PYTHONPATH

mkdir -p logs

echo "Starting checkpoint for snapshot ${SNAP}  (array task ${SLURM_ARRAY_TASK_ID})"
date

python $HOME/DirectSHAM/pipeline/make_checkpoint_snap.py \
    --snapshot    ${SNAP}        \
    --reg         NS             \
    --sig_lgv_ref  ${SIG_LGV_REF}  \
    --sig_lgv_slope ${SIG_LGV_SLOPE} \
    --f_colour    ${F_COLOUR}    \

echo "Finished snapshot ${SNAP}"
date
