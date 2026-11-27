#!/bin/bash
# run_pipeline.sh — Build a cut-sky mock and measure its clustering for the
# Vpeak-dependent scatter parametrisation:
#   sig_lgv = SIG_LGV_REF + SIG_LGV_SLOPE * (log10(Vpeak) - 2.6)
# plus the assembly-history boost alpha_v (SIGMA_MIN = 0.035 hardwired).
#
# Stages (default: all):
#   checkpoint  — compute node required (~33 min, needs --mem=400GB)
#   mock        — make_mock N then S, split NGC/SGC (login node OK)
#   clustering  — mock clustering for the 5 luminosity samples (compute node)
#
# Compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi --mem=400GB --time=04:00:00
#
# Examples:
#   bash run_pipeline.sh --sig_lgv_ref 0.21 --sig_lgv_slope 0.36
#   bash run_pipeline.sh --sig_lgv_ref 0.21 --sig_lgv_slope 0.36 --stage mock
#   bash run_pipeline.sh --sig_lgv_ref 0.21 --sig_lgv_slope 0.36 --stage clustering
#
# Output: $DATA_DIR/<DATA_DATE>/N_neigh_lum50/
#           seed{OBS_SEED}_sig_lgv{REF}_{SLOPE}_fc{F_COLOUR}_al{ALPHA}/
# Use two-decimal parameter values: directory names are formatted with %.2f.

set -e

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/DirectSHAM/pipeline:$PYTHONPATH

cd ~/DirectSHAM/pipeline

# --- Defaults: current fiducial model (16thNov) ---
OBS_SEED=0
SIG_LGV_REF=0.18
SIG_LGV_SLOPE=0.30
F_COLOUR=0.00
ALPHA=0.02
STAGE=all

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --obs_seed)      OBS_SEED=$2;      shift 2 ;;
        --sig_lgv_ref)   SIG_LGV_REF=$2;   shift 2 ;;
        --sig_lgv_slope) SIG_LGV_SLOPE=$2; shift 2 ;;
        --f_colour)      F_COLOUR=$2;      shift 2 ;;
        --alpha)         ALPHA=$2;         shift 2 ;;
        --stage)         STAGE=$2;         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case $STAGE in
    all|checkpoint|mock|clustering) ;;
    *) echo "Unknown stage: $STAGE (use all|checkpoint|mock|clustering)"; exit 1 ;;
esac

# Every script gets the full parameter set, so all stages agree on the
# model directory. (make_checkpoint.py defaults to --alpha 0.0, not 0.02.)
PARAMS="--obs_seed $OBS_SEED --sig_lgv_ref $SIG_LGV_REF --sig_lgv_slope $SIG_LGV_SLOPE --f_colour $F_COLOUR --alpha $ALPHA"

# 5 luminosity-threshold samples: mr_cut zmin zmax
SAMPLES=(
    "-21.5 0.02 0.42"
    "-20.5 0.02 0.28"
    "-19.5 0.02 0.19"
    "-18.5 0.02 0.12"
    "-17.5 0.02 0.08"
)

echo "============================================================"
echo "stage=$STAGE"
echo "$PARAMS"
echo "============================================================"

if [[ $STAGE == all || $STAGE == checkpoint ]]; then
    echo "--- Making checkpoint (N and S) ---"
    python make_checkpoint.py --reg NS $PARAMS
fi

if [[ $STAGE == all || $STAGE == mock ]]; then
    echo "--- Making mock (N must come before S) ---"
    python make_mock.py --reg N $PARAMS
    python make_mock.py --reg S $PARAMS

    echo "--- Splitting into NGC/SGC ---"
    python split_ngcsgc.py $PARAMS
fi

if [[ $STAGE == all || $STAGE == clustering ]]; then
    for SAMPLE in "${SAMPLES[@]}"; do
        read MR ZMIN ZMAX <<< "$SAMPLE"
        echo "------------------------------------------------------------"
        echo "Sample: Mr < $MR  z = $ZMIN - $ZMAX"
        echo "------------------------------------------------------------"
        python measure_mock_clustering.py $PARAMS \
            --mr_cut $MR --zmin $ZMIN --zmax $ZMAX
    done
fi

echo "============================================================"
echo "Stage '$STAGE' complete."
echo "============================================================"
