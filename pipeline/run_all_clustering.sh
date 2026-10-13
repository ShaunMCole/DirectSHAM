#!/bin/bash
# run_all_clustering.sh — Measure data and mock clustering for all 5 BGS samples.
#
# Must be run on a compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi --time=04:00:00
#
# Then:
#   cd ~/DirectSHAM/pipeline
#   bash run_all_clustering.sh

set -e

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/ShaunColeDESI:$PYTHONPATH

cd ~/DirectSHAM/pipeline

OBS_SEED=0
SIG_LGV=0.04
NJACK=60

# 5 samples: mr_cut zmin zmax
SAMPLES=(
    "-22.0 0.02 0.50"
    "-21.5 0.02 0.42"
    "-21.0 0.02 0.34"
    "-20.5 0.02 0.28"
    "-20.0 0.02 0.23"
)

for SAMPLE in "${SAMPLES[@]}"; do
    read MR ZMIN ZMAX <<< "$SAMPLE"
    echo "============================================================"
    echo "Sample: Mr < $MR  z = $ZMIN - $ZMAX"
    echo "============================================================"

    echo "--- Data clustering ---"
    python measure_clustering.py \
        --mr_cut $MR --zmin $ZMIN --zmax $ZMAX --njack $NJACK

    echo "--- Mock clustering ---"
    python measure_mock_clustering.py \
        --obs_seed $OBS_SEED --sig_lgv $SIG_LGV \
        --mr_cut $MR --zmin $ZMIN --zmax $ZMAX
done

echo "============================================================"
echo "All samples done."
echo "============================================================"
