#!/bin/bash
# run_cigale_node2.sh — z=0.107-0.344
cd ~/ShaunColeDESI
SCRIPT=~/DirectSHAM/pipeline/measure_clustering_mstar.py
SIG_REF=0.18
SIG_SLOPE=0.30

run() {
    local MODE=$1 MMIN=$2 MMAX=$3 ZMIN=$4 ZMAX=$5
    echo "=== $MODE  Mcig ${MMIN}-${MMAX}  z ${ZMIN}-${ZMAX} ==="
    python $SCRIPT --mode $MODE \
        --sample_type cigale \
        --mstar_cut $MMIN --mstar_max $MMAX \
        --zmin $ZMIN --zmax $ZMAX \
        --sig_lgv_ref $SIG_REF --sig_lgv_slope $SIG_SLOPE
}

# z=0.107-0.162: 3 mass bins
run data 10.0 10.5  0.107 0.162
run mock 10.0 10.5  0.107 0.162
run data 10.5 11.0  0.107 0.162
run mock 10.5 11.0  0.107 0.162
run data 11.0 12.0  0.107 0.162
run mock 11.0 12.0  0.107 0.162

# z=0.162-0.240: 2 mass bins
run data 10.5 11.0  0.162 0.240
run mock 10.5 11.0  0.162 0.240
run data 11.0 12.0  0.162 0.240
run mock 11.0 12.0  0.162 0.240

# z=0.240-0.344: 1 mass bin
run data 11.0 12.0  0.240 0.344
run mock 11.0 12.0  0.240 0.344

echo "=== Node 2 complete ==="
