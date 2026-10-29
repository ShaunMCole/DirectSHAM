#!/bin/bash
# run_mstar_clustering.sh
# =======================
# Measure two-point clustering for the five stellar mass threshold samples
# (CIGALE and FSF) for both data and mock catalogues.
#
# Must be run on a compute node:
#   salloc --nodes=1 --qos=interactive --constraint=cpu \
#          --account=desi --mem=128GB --time=04:00:00
#
# Usage:
#   bash pipeline/run_mstar_clustering.sh             # data + mock
#   bash pipeline/run_mstar_clustering.sh data        # data only
#   bash pipeline/run_mstar_clustering.sh mock        # mock only
#
# Model parameters (edit these to change the mock):
SIG_LGV_REF=0.18
SIG_LGV_SLOPE=0.30
F_COLOUR=0.00
ALPHA=0.00
OBS_SEED=0
NJACK=60

# Stellar mass thresholds and matching zmax values.
# Each row: CIGALE_CUT  FSF_CUT  ZMAX
SAMPLES=(
    "11.10  11.45  0.42"
    "10.60  10.95  0.34"
    "10.10  10.45  0.27"
    " 9.50   9.85  0.21"
    " 8.90   9.25  0.17"
)

SAMPLES=(
    "11.10  11.45  0.42"
    "10.10  10.45  0.27"
)

ZMIN=0.02
MODE=${1:-"both"}   # data, mock, or both (default)

set -e

# Source DESI environment if not already loaded
if ! command -v xirunpc.py &> /dev/null; then
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
fi
export PYTHONPATH=":$SCRATCH/LSS/py:$HOME/DirectSHAM/pipeline:$PYTHONPATH"

cd ~/DirectSHAM

echo "============================================================"
echo "run_mstar_clustering.sh"
echo "mode=${MODE}  sig_lgv_ref=${SIG_LGV_REF}  sig_lgv_slope=${SIG_LGV_SLOPE}"
echo "f_colour=${F_COLOUR}  alpha=${ALPHA}  njack=${NJACK}"
echo "============================================================"

run_data() {
    local sample_type=$1
    local mstar_cut=$2
    local zmax=$3
    echo ""
    echo "--- DATA  sample_type=${sample_type}  mstar_cut=${mstar_cut}  zmax=${zmax} ---"
    python pipeline/measure_clustering_mstar.py \
        --mode data \
        --sample_type "${sample_type}" \
        --mstar_cut "${mstar_cut}" \
        --zmin "${ZMIN}" \
        --zmax "${zmax}" \
        --njack "${NJACK}"
}

run_mock() {
    local sample_type=$1
    local mstar_cut=$2
    local zmax=$3
    echo ""
    echo "--- MOCK  sample_type=${sample_type}  mstar_cut=${mstar_cut}  zmax=${zmax} ---"
    python pipeline/measure_clustering_mstar.py \
        --mode mock \
        --sample_type "${sample_type}" \
        --mstar_cut "${mstar_cut}" \
        --zmin "${ZMIN}" \
        --zmax "${zmax}" \
        --sig_lgv_ref "${SIG_LGV_REF}" \
        --sig_lgv_slope "${SIG_LGV_SLOPE}" \
        --f_colour "${F_COLOUR}" \
        --alpha "${ALPHA}" \
        --obs_seed "${OBS_SEED}"
}

for sample in "${SAMPLES[@]}"; do
    read -r cig_cut fsf_cut zmax <<< "${sample}"

    if [[ "${MODE}" == "data" || "${MODE}" == "both" ]]; then
        run_data cigale "${cig_cut}" "${zmax}"
        run_data fsf    "${fsf_cut}" "${zmax}"
    fi

    if [[ "${MODE}" == "mock" || "${MODE}" == "both" ]]; then
        run_mock cigale "${cig_cut}" "${zmax}"
        run_mock fsf    "${fsf_cut}" "${zmax}"
    fi

done

echo ""
echo "============================================================"
echo "run_mstar_clustering.sh complete."
echo "============================================================"
