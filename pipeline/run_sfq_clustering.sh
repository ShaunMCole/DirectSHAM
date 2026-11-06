#!/usr/bin/env bash
# run_sfq_clustering.sh — measure clustering for SF and passive subsamples
# of each CIGALE stellar mass threshold sample.
#
# Must be run on a compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu \
#          --account=desi --mem=128GB --time=08:00:00
#
# Usage:
#   bash pipeline/run_sfq_clustering.sh [data|mock|both]
#
# Produces sample tags: Mcig{cut}_z{zmin}-{zmax}_{sf|passive}

set -euo pipefail

MODE=${1:-both}
SIG_LGV_REF=0.18
SIG_LGV_SLOPE=0.30

# CIGALE cut, zmax pairs
SAMPLES=(
    "11.10  0.42"
    "10.60  0.34"
    "10.10  0.27"
    " 9.50  0.21"
    " 8.90  0.17"
)

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=":$SCRATCH/LSS/py:$PYTHONPATH"
cd ~/DirectSHAM

for SAMPLE in "${SAMPLES[@]}"; do
    read CIG_CUT ZMAX <<< "$SAMPLE"
    for SFQ in sf passive; do
        echo ""
        echo "========================================"
        echo "CIGALE ${CIG_CUT}  z0.02-${ZMAX}  sfq=${SFQ}"
        echo "========================================"

        if [[ "$MODE" == "data" || "$MODE" == "both" ]]; then
            echo "--- DATA ---"
            python pipeline/measure_clustering_mstar.py \
                --mode data \
                --sample_type cigale \
                --mstar_cut ${CIG_CUT} \
                --zmin 0.02 --zmax ${ZMAX} \
                --sfq ${SFQ} \
                --sig_lgv_ref ${SIG_LGV_REF} \
                --sig_lgv_slope ${SIG_LGV_SLOPE}
        fi

        if [[ "$MODE" == "mock" || "$MODE" == "both" ]]; then
            echo "--- MOCK ---"
            python pipeline/measure_clustering_mstar.py \
                --mode mock \
                --sample_type cigale \
                --mstar_cut ${CIG_CUT} \
                --zmin 0.02 --zmax ${ZMAX} \
                --sfq ${SFQ} \
                --sig_lgv_ref ${SIG_LGV_REF} \
                --sig_lgv_slope ${SIG_LGV_SLOPE}
        fi

    done
done

echo ""
echo "All sfq clustering runs complete."
