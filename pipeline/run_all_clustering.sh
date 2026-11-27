#!/bin/bash
# run_all_clustering.sh — Measure data and mock clustering for all 5 BGS
# luminosity-threshold samples.
#
# Must be run on a compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi --mem=400GB --time=04:00:00
#
# Then:
#   cd ~/DirectSHAM/pipeline
#   bash run_all_clustering.sh [--obs_seed N] [--sig_lgv_ref R] [--sig_lgv_slope S]
#                              [--f_colour F] [--alpha A] [--njack J]
#                              [--skip_data] [--skip_mock]
#
# Defaults are the current fiducial model (16thNov; sig_lgv 0.24, 0.42; alpha_v = 0.02).
# Data clustering is skipped automatically if it already exists under
# DATA_DATE_DIR/clustering/data/ (DATA_DATE_DIR taken from config.py).

set -e

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/DirectSHAM/pipeline:$PYTHONPATH

cd ~/DirectSHAM/pipeline

# --- Defaults ---
OBS_SEED=0
SIG_LGV_REF=0.24
SIG_LGV_SLOPE=0.42
F_COLOUR=0.00
ALPHA=0.02
NJACK=60
SKIP_DATA=0
SKIP_MOCK=0

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --obs_seed)      OBS_SEED=$2;       shift 2 ;;
        --sig_lgv_ref)   SIG_LGV_REF=$2;    shift 2 ;;
        --sig_lgv_slope) SIG_LGV_SLOPE=$2;  shift 2 ;;
        --f_colour)      F_COLOUR=$2;       shift 2 ;;
        --alpha)         ALPHA=$2;          shift 2 ;;
        --njack)         NJACK=$2;          shift 2 ;;
        --skip_data)     SKIP_DATA=1;       shift 1 ;;
        --skip_mock)     SKIP_MOCK=1;       shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Data clustering location follows config.py (same path measure_clustering.py writes to)
DATA_DATE_DIR=$(python -c "from config import DATA_DATE_DIR; print(DATA_DATE_DIR)")

echo "obs_seed=$OBS_SEED  sig_lgv_ref=$SIG_LGV_REF  sig_lgv_slope=$SIG_LGV_SLOPE  f_colour=$F_COLOUR  alpha=$ALPHA"
echo "data clustering dir: $DATA_DATE_DIR/clustering/data"

# 5 samples: mr_cut zmin zmax
SAMPLES=(
    "-21.5 0.02 0.42"
    "-20.5 0.02 0.28"
    "-19.5 0.02 0.19"
    "-18.5 0.02 0.12"
    "-17.5 0.02 0.08"
)

for SAMPLE in "${SAMPLES[@]}"; do
    read MR ZMIN ZMAX <<< "$SAMPLE"
    echo "============================================================"
    echo "Sample: Mr < $MR  z = $ZMIN - $ZMAX"
    echo "============================================================"

    if [[ $SKIP_DATA -eq 0 ]]; then
        DATA_OUTDIR="$DATA_DATE_DIR/clustering/data/Mr${MR}_z${ZMIN}-${ZMAX}"
        if [[ -d "$DATA_OUTDIR/smu" ]] && ls "$DATA_OUTDIR"/smu/*.npy 1>/dev/null 2>&1; then
            echo "--- Data clustering already exists, skipping ---"
        else
            echo "--- Data clustering ---"
            python measure_clustering.py \
                --mr_cut $MR --zmin $ZMIN --zmax $ZMAX --njack $NJACK
        fi
    else
        echo "--- Data clustering skipped (--skip_data) ---"
    fi

    if [[ $SKIP_MOCK -eq 0 ]]; then
        echo "--- Mock clustering ---"
        python measure_mock_clustering.py \
            --obs_seed $OBS_SEED \
            --sig_lgv_ref $SIG_LGV_REF \
            --sig_lgv_slope $SIG_LGV_SLOPE \
            --f_colour $F_COLOUR \
            --alpha $ALPHA \
            --mr_cut $MR --zmin $ZMIN --zmax $ZMAX
    else
        echo "--- Mock clustering skipped (--skip_mock) ---"
    fi
done

echo "============================================================"
echo "All samples done."
echo "============================================================"
