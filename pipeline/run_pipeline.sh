#!/bin/bash
# run_pipeline.sh — Run mock construction and clustering for the
# Vpeak-dependent scatter parametrisation:
#   sig_lgv = SIG_LGV_REF + SIG_LGV_SLOPE * (log10(Vpeak) - 2.6)
#
# Must be run on a compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi --time=04:00:00
#
# Then:
#   cd ~/DirectSHAM/pipeline
#   bash run_pipeline.sh

set -e

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/DirectSHAM/pipeline:$PYTHONPATH

cd ~/DirectSHAM/pipeline

OBS_SEED=0
SIG_LGV_REF=0.14
SIG_LGV_SLOPE=0.30

# 5 samples: mr_cut zmin zmax
SAMPLES=(
    "-22.0 0.02 0.50"
    "-21.5 0.02 0.42"
    "-21.0 0.02 0.34"
    "-20.5 0.02 0.28"
    "-20.0 0.02 0.23"
)

echo "============================================================"
echo "sig_lgv_ref=${SIG_LGV_REF}  sig_lgv_slope=${SIG_LGV_SLOPE}"
echo "============================================================"

echo "--- Making checkpoint (N and S) ---"
python make_checkpoint.py --reg NS --obs_seed $OBS_SEED \
    --sig_lgv_ref $SIG_LGV_REF --sig_lgv_slope $SIG_LGV_SLOPE

echo "--- Making mock (N must come before S) ---"
python make_mock.py --reg N --obs_seed $OBS_SEED \
    --sig_lgv_ref $SIG_LGV_REF --sig_lgv_slope $SIG_LGV_SLOPE
python make_mock.py --reg S --obs_seed $OBS_SEED \
    --sig_lgv_ref $SIG_LGV_REF --sig_lgv_slope $SIG_LGV_SLOPE

echo "--- Splitting into NGC/SGC ---"
python split_ngcsgc.py --obs_seed $OBS_SEED \
    --sig_lgv_ref $SIG_LGV_REF --sig_lgv_slope $SIG_LGV_SLOPE

for SAMPLE in "${SAMPLES[@]}"; do
    read MR ZMIN ZMAX <<< "$SAMPLE"
    echo "------------------------------------------------------------"
    echo "Sample: Mr < $MR  z = $ZMIN - $ZMAX"
    echo "------------------------------------------------------------"

    echo "--- Measuring mock clustering ---"
    python measure_mock_clustering.py \
        --obs_seed $OBS_SEED \
        --sig_lgv_ref $SIG_LGV_REF --sig_lgv_slope $SIG_LGV_SLOPE \
        --mr_cut $MR --zmin $ZMIN --zmax $ZMAX
done

echo "============================================================"
echo "Pipeline complete."
echo "============================================================"
