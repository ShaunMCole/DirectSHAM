#!/bin/bash
# run_cigale_node1.sh — z=0.027-0.107 (first cell already done)
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

# z=0.027-0.043: 4 remaining mass bins (8.5-9.0 already done)
run data  9.0  9.5  0.027 0.043
run mock  9.0  9.5  0.027 0.043
run data  9.5 10.0  0.027 0.043
run mock  9.5 10.0  0.027 0.043
run data 10.0 10.5  0.027 0.043
run mock 10.0 10.5  0.027 0.043
run data 10.5 11.0  0.027 0.043
run mock 10.5 11.0  0.027 0.043

# z=0.043-0.068: 4 mass bins
run data  9.0  9.5  0.043 0.068
run mock  9.0  9.5  0.043 0.068
run data  9.5 10.0  0.043 0.068
run mock  9.5 10.0  0.043 0.068
run data 10.0 10.5  0.043 0.068
run mock 10.0 10.5  0.043 0.068
run data 10.5 11.0  0.043 0.068
run mock 10.5 11.0  0.043 0.068

# z=0.068-0.107: 4 mass bins
run data  9.5 10.0  0.068 0.107
run mock  9.5 10.0  0.068 0.107
run data 10.0 10.5  0.068 0.107
run mock 10.0 10.5  0.068 0.107
run data 10.5 11.0  0.068 0.107
run mock 10.5 11.0  0.068 0.107
run data 11.0 12.0  0.068 0.107
run mock 11.0 12.0  0.068 0.107

echo "=== Node 1 complete ==="
