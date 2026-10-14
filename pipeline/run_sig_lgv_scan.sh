#!/bin/bash
# run_sig_lgv_scan.sh — Run mock construction and clustering for additional
# sig_lgv values across all 5 magnitude-limited samples.
#
# Must be run on a compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi --time=04:00:00
#
# Then:
#   cd ~/DirectSHAM/pipeline
#   bash run_sig_lgv_scan.sh

set -e

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/ShaunColeDESI:$PYTHONPATH

cd ~/DirectSHAM/pipeline

OBS_SEED=0

# sig_lgv values to scan
SIG_LGV_VALUES=(0.08)

# 5 samples: mr_cut zmin zmax 
SAMPLES=(
    "-22.0 0.02 0.50"
    "-21.5 0.02 0.42"
    "-21.0 0.02 0.34"
    "-20.5 0.02 0.28"
    "-20.0 0.02 0.23"
)

for SIG_LGV in "${SIG_LGV_VALUES[@]}"; do
    echo "============================================================"
    echo "sig_lgv=${SIG_LGV}"
    echo "============================================================"

    echo "--- Making checkpoint (N and S) ---"
    python make_checkpoint.py --reg NS --obs_seed $OBS_SEED --sig_lgv $SIG_LGV

    echo "--- Making mock (N must come before S) ---"
    python make_mock.py --reg N --obs_seed $OBS_SEED --sig_lgv $SIG_LGV
    python make_mock.py --reg S --obs_seed $OBS_SEED --sig_lgv $SIG_LGV

    echo "--- Splitting into NGC/SGC ---"
    python split_ngcsgc.py --obs_seed $OBS_SEED --sig_lgv $SIG_LGV

    for SAMPLE in "${SAMPLES[@]}"; do
        read MR ZMIN ZMAX <<< "$SAMPLE"
        echo "------------------------------------------------------------"
        echo "Sample: Mr < $MR  z = $ZMIN - $ZMAX  sig_lgv=$SIG_LGV"
        echo "------------------------------------------------------------"

        echo "--- Measuring mock clustering ---"
        python measure_mock_clustering.py \
            --obs_seed $OBS_SEED --sig_lgv $SIG_LGV \
            --mr_cut $MR --zmin $ZMIN --zmax $ZMAX
    done
done

echo "============================================================"
echo "sig_lgv scan complete."
echo "============================================================"
