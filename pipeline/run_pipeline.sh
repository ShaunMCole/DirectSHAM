#!/bin/bash
# run_pipeline.sh — Orchestrate the full DirectSHAM mock pipeline for one or
# more values of the Vpeak scatter parameter (sig_lgv).
#
# For each scatter value the script:
#   1. Submits make_checkpoint.py for N and S (expensive, SLURM)
#   2. Chains make_mock.py for N then S (login node, after checkpoint)
#   3. Chains split_ngcsgc.py           (login node, after both mocks)
#   4. Chains measure_mock_clustering.py (SLURM, after split)
#
# The data clustering (measure_clustering.py) is submitted once, independently
# of the scatter loop, since it does not depend on the mocks.
#
# Usage
# -----
#   bash run_pipeline.sh [options] sig_lgv [sig_lgv ...]
#
# Options
#   --obs_seed  N     Observer position seed (default: 0)
#   --mr_cut    X     Absolute magnitude cut (default: -21.5)
#   --zmin      X     Minimum redshift       (default: 0.02)
#   --zmax      X     Maximum redshift       (default: 0.42)
#   --njack     N     Jackknife regions for data xi (default: 0)
#   --no_data         Skip the data clustering measurement
#   --dry_run         Print commands without submitting
#
# Examples
# --------
#   # One scatter value:
#   bash run_pipeline.sh 0.10
#
#   # Several scatter values:
#   bash run_pipeline.sh 0.00 0.05 0.10 0.15 0.20
#
#   # Custom magnitude cut, preview commands only:
#   bash run_pipeline.sh --mr_cut -21.0 --dry_run 0.10 0.15

set -euo pipefail

# =============================================================================
# Defaults
# =============================================================================
OBS_SEED=0
MR_CUT=-21.5
ZMIN=0.02
ZMAX=0.42
NJACK=0
RUN_DATA=true
DRY_RUN=false

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PIPELINE_DIR"
mkdir -p logs

# =============================================================================
# Parse arguments
# =============================================================================
SIG_LGV_LIST=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --obs_seed)  OBS_SEED="$2";  shift 2 ;;
        --mr_cut)    MR_CUT="$2";    shift 2 ;;
        --zmin)      ZMIN="$2";      shift 2 ;;
        --zmax)      ZMAX="$2";      shift 2 ;;
        --njack)     NJACK="$2";     shift 2 ;;
        --no_data)   RUN_DATA=false; shift   ;;
        --dry_run)   DRY_RUN=true;   shift   ;;
        --*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            SIG_LGV_LIST+=("$1"); shift ;;
    esac
done

if [[ ${#SIG_LGV_LIST[@]} -eq 0 ]]; then
    echo "Usage: bash run_pipeline.sh [options] sig_lgv [sig_lgv ...]" >&2
    exit 1
fi

# =============================================================================
# Helper: submit or print
# =============================================================================
submit() {
    # submit SCRIPT [--dependency=afterok:JOBID]
    # Returns the job ID in REPLY.
    local script="$1"; shift
    local dep_arg="${1:-}"
    if $DRY_RUN; then
        echo "[DRY RUN] sbatch ${dep_arg} ${script}"
        REPLY=DRYRUN
    else
        REPLY=$(sbatch ${dep_arg} "$script" | awk '{print $NF}')
        echo "Submitted $script → job $REPLY"
    fi
}

run_login() {
    # run_login COMMAND [--dependency=JOBID]
    # Runs a command on the login node, but waits for a SLURM job to finish
    # first if a dependency is given.
    local cmd="$1"; shift
    local dep_jobid="${1:-}"

    if [[ -n "$dep_jobid" && "$dep_jobid" != "DRYRUN" ]]; then
        echo "Waiting for job $dep_jobid before running: $cmd"
        if ! $DRY_RUN; then
            # Poll until the job is no longer in the queue.
            while squeue -j "$dep_jobid" -h 2>/dev/null | grep -q "$dep_jobid"; do
                sleep 30
            done
        fi
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] $cmd"
    else
        echo "Running: $cmd"
        eval "$cmd"
    fi
}

# =============================================================================
# Source the DESI environment (needed for login-node Python steps)
# =============================================================================
if ! $DRY_RUN; then
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    export PYTHONPATH=:$SCRATCH/LSS/py:$PYTHONPATH
fi

# =============================================================================
# Step 0: Data clustering (independent of scatter loop)
# =============================================================================
DATA_JOB=""
if $RUN_DATA; then
    echo ""
    echo "========================================"
    echo "Submitting data clustering measurement"
    echo "========================================"
    export MR_CUT ZMIN ZMAX NJACK
    submit slurm_measure_clustering.sh
    DATA_JOB="$REPLY"
fi

# =============================================================================
# Scatter loop
# =============================================================================
for SIG_LGV in "${SIG_LGV_LIST[@]}"; do
    echo ""
    echo "========================================"
    echo "sig_lgv=${SIG_LGV}  obs_seed=${OBS_SEED}"
    echo "========================================"

    # -------------------------------------------------------------------------
    # Step 1: Checkpoint — N and S submitted in parallel
    # -------------------------------------------------------------------------
    echo "--- Step 1: Checkpoints ---"

    export REG=N OBS_SEED SIG_LGV
    submit slurm_checkpoint.sh
    CKPT_N_JOB="$REPLY"

    export REG=S
    submit slurm_checkpoint.sh
    CKPT_S_JOB="$REPLY"

    # -------------------------------------------------------------------------
    # Step 2: make_mock N (depends on checkpoint N)
    # Step 3: make_mock S (depends on checkpoint S AND mock N for NMULT)
    # -------------------------------------------------------------------------
    echo "--- Step 2/3: Mocks ---"

    MOCK_N_CMD="python make_mock.py --reg N --obs_seed ${OBS_SEED} --sig_lgv ${SIG_LGV}"
    run_login "$MOCK_N_CMD" "$CKPT_N_JOB"

    # South also needs North mock to exist (for NMULT matching), so we wait
    # for both the checkpoint S job and the mock N step.
    MOCK_S_CMD="python make_mock.py --reg S --obs_seed ${OBS_SEED} --sig_lgv ${SIG_LGV}"
    run_login "$MOCK_S_CMD" "$CKPT_S_JOB"

    # -------------------------------------------------------------------------
    # Step 4: NGC/SGC split (depends on both mocks)
    # -------------------------------------------------------------------------
    echo "--- Step 4: NGC/SGC split ---"

    SPLIT_CMD="python split_ngcsgc.py --obs_seed ${OBS_SEED} --sig_lgv ${SIG_LGV} --mocks --formock"
    run_login "$SPLIT_CMD"

    # -------------------------------------------------------------------------
    # Step 5: Mock clustering measurement (SLURM)
    # -------------------------------------------------------------------------
    echo "--- Step 5: Mock clustering ---"

    export OBS_SEED SIG_LGV MR_CUT ZMIN ZMAX
    submit slurm_measure_mock.sh
    MOCK_XI_JOB="$REPLY"

    echo "Mock clustering job: $MOCK_XI_JOB"

done  # end scatter loop

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Pipeline submitted."
echo "sig_lgv values: ${SIG_LGV_LIST[*]}"
echo "obs_seed:       ${OBS_SEED}"
echo "mr_cut:         ${MR_CUT}"
echo "z range:        [${ZMIN}, ${ZMAX}]"
if [[ -n "$DATA_JOB" ]]; then
    echo "Data xi job:    ${DATA_JOB}"
fi
echo ""
echo "When all jobs complete, compare results with:"
echo "  python compare_clustering.py \\"
echo "      --data_file  \$DATA_DIR/clustering/data/... \\"
echo "      --mock_files \$DATA_DIR/clustering/mocks/*/... \\"
echo "      --smin 5 --smax 50"
echo "========================================"
