#!/bin/bash
# run_cigale_clustering.sh
# Run data and mock clustering for all CIGALE stellar mass bin x redshift
# bin combinations requested by Shun Saito for the KP HOD paper.
#
# Must be run on a compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu \
#          --account=desi --mem=400GB --time=04:00:00
#   source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#   bash pipeline/run_cigale_clustering.sh
#
# 19 cells x 2 (data+mock) = 38 runs.
# Split into two jobs if needed by commenting out the second half.

cd ~/DirectSHAM
SCRIPT=~/DirectSHAM/pipeline/measure_clustering_mstar.py
SIG_REF=0.18
SIG_SLOPE=0.30
ALPHA=0.02

run() {
    local MODE=$1 MMIN=$2 MMAX=$3 ZMIN=$4 ZMAX=$5
    echo "=== $MODE  Mcig ${MMIN}-${MMAX}  z ${ZMIN}-${ZMAX} ==="
    python $SCRIPT --mode $MODE \
        --sample_type cigale \
        --mstar_cut $MMIN --mstar_max $MMAX \
        --zmin $ZMIN --zmax $ZMAX \
        --sig_lgv_ref $SIG_REF --sig_lgv_slope $SIG_SLOPE \
        --alpha $ALPHA
}

# z=0.027-0.043: 5 mass bins
run data  8.5  9.0  0.027 0.043
run mock  8.5  9.0  0.027 0.043
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

echo "=== All clustering runs complete ==="
