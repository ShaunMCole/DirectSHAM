#!/usr/bin/env bash
# propagate_stellar_masses.sh
# ============================
# Propagate stellar mass and SFR columns from the augmented master BGS
# catalogue to ALL files that need them:
#
#   - BGS data NGC/SGC subsets
#   - fordata randoms NGC/SGC
#   - formock randoms NGC/SGC
#   - mock NGC/SGC (cut-sky)
#
# The fordata and formock NGC randoms are ~50M rows and will OOM on a login
# node.  Always run this script on a compute node:
#
#   salloc --nodes=1 --qos=interactive --constraint=cpu \
#          --account=desi --mem=512GB --time=04:00:00
#   source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#   export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/ShaunColeDESI:$PYTHONPATH
#   bash ~/DirectSHAM/pipeline/propagate_stellar_masses.sh
#
# The script processes files ONE AT A TIME so that memory is freed between
# files.  This avoids holding multiple large tables in memory simultaneously.
#
# To propagate a different data date or model, set DATA_DATE and MODEL_DIR
# before running:
#
#   DATA_DATE=16thNov MODEL_DIR=N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.02 \
#       bash ~/DirectSHAM/pipeline/propagate_stellar_masses.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment variables if needed
# ---------------------------------------------------------------------------
DATA_DIR="/global/cfs/cdirs/desi/users/smcole/DirectSHAM"
DATA_DATE="${DATA_DATE:-16thNov}"
MODEL_DIR="${MODEL_DIR:-N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.02}"

DATE_DIR="${DATA_DIR}/${DATA_DATE}"
MODEL_PATH="${DATE_DIR}/${MODEL_DIR}"
MASTER="${DATE_DIR}/BGS_BRIGHT_${DATA_DATE}_augmented.fits"
SCRIPT="$(dirname "$0")/propagate_stellar_masses.py"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ ! -f "${MASTER}" ]]; then
    echo "ERROR: master catalogue not found: ${MASTER}"
    exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: propagate_stellar_masses.py not found: ${SCRIPT}"
    exit 1
fi

echo "============================================================"
echo "propagate_stellar_masses.sh"
echo "DATA_DATE : ${DATA_DATE}"
echo "MODEL_DIR : ${MODEL_DIR}"
echo "Master    : ${MASTER}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Files to process — in order from smallest to largest to catch errors early
# ---------------------------------------------------------------------------
# NOTE: randoms MUST receive the same stellar mass cuts as data and mock.
# The formock and fordata randoms are clones of the data galaxies and carry
# TARGETID, so the join works exactly as for the data files.  Skipping the
# randoms produces wrong dN/dz for stellar-mass-selected samples and wrong
# clustering amplitudes.

FILES=(
    # --- Data subsets (small, process first) ---
    "${DATE_DIR}/BGS_BRIGHT_${DATA_DATE}_NGC.fits"
    "${DATE_DIR}/BGS_BRIGHT_${DATA_DATE}_SGC.fits"

    # --- Mock catalogues (medium) ---
    "${MODEL_PATH}/mock_NGC.fits"
    "${MODEL_PATH}/mock_SGC.fits"

    # --- fordata randoms SGC (medium) ---
    "${DATE_DIR}/rancat_fordata_SGC.fits"

    # --- formock randoms SGC (medium) ---
    "${DATE_DIR}/rancat_formock_SGC.fits"

    # --- fordata randoms NGC (~50M rows — needs 512GB node) ---
    "${DATE_DIR}/rancat_fordata_NGC.fits"

    # --- formock randoms NGC (~50M rows — needs 512GB node) ---
    "${DATE_DIR}/rancat_formock_NGC.fits"
)

# ---------------------------------------------------------------------------
# Process each file individually so memory is freed between files
# ---------------------------------------------------------------------------
N_OK=0
N_FAIL=0
N_SKIP=0

for f in "${FILES[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "WARNING: file not found, skipping: ${f}"
        (( N_SKIP++ )) || true
        continue
    fi
    echo ""
    echo "------------------------------------------------------------"
    echo "Processing: $(basename ${f})"
    echo "------------------------------------------------------------"
    python "${SCRIPT}" \
        --master "${MASTER}" \
        --files  "${f}" \
    && (( N_OK++ )) || { echo "ERROR: failed for ${f}"; (( N_FAIL++ )) || true; }
done

echo ""
echo "============================================================"
echo "Done: ${N_OK} succeeded, ${N_FAIL} failed, ${N_SKIP} skipped"
echo "============================================================"
