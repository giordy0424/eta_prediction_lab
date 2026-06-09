#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${BASE_DIR}/data"
PHASE2_DIR="${BASE_DIR}/phase2_dataset"
PHASE3_DIR="${BASE_DIR}/phase3_training"
PHASE4_DIR="${BASE_DIR}/phase4_evaluation"
MODELS_DIR="${BASE_DIR}/models"

KAGGLE_CSV="${DATA_DIR}/train.csv"
PREPARED_CSV="${DATA_DIR}/_prepared.csv"
TRAIN_OUT="${DATA_DIR}/ultimate_train.csv"
TEST_OUT="${DATA_DIR}/ultimate_test.csv"

usage() {
    echo "Usage: $0 [--phase1|--phase2|--phase3|--phase4|--all]"
    exit 1
}

phase1() {
    echo "=== Phase 1: OSRM Server ==="
    if [ ! -d "${BASE_DIR}/osrm_data" ]; then
        mkdir -p "${BASE_DIR}/osrm_data"
    fi
    bash "${BASE_DIR}/phase1_osrm/setup_server.sh"
}

phase2() {
    echo "=== Phase 2: Dataset ==="
    if [ ! -f "$KAGGLE_CSV" ]; then
        echo "ERROR: $KAGGLE_CSV not found. Download from Kaggle first."
        exit 1
    fi
    python "${PHASE2_DIR}/prepare_dataset.py" \
        --input "$KAGGLE_CSV" \
        --mask "${PHASE2_DIR}/mask.geojson" \
        --output "$PREPARED_CSV"
    python "${PHASE2_DIR}/enrich_and_split.py" \
        --input "$PREPARED_CSV" \
        --output-train "$TRAIN_OUT" \
        --output-test "$TEST_OUT"
    echo "Phase 2 complete:"
    echo "  Train: $TRAIN_OUT"
    echo "  Test:  $TEST_OUT"
}

phase3() {
    echo "=== Phase 3: Training (optional) ==="
    if [ ! -f "$TRAIN_OUT" ]; then
        echo "ERROR: $TRAIN_OUT not found. Run --phase2 first."
        exit 1
    fi
    echo "Training naive models..."
    python "${PHASE3_DIR}/train_naive.py" --data "$TRAIN_OUT" --output-dir "${MODELS_DIR}/naive"
    echo "Training final models..."
    python "${PHASE3_DIR}/train_final.py" --data "$TRAIN_OUT" --output-dir "${MODELS_DIR}/final"
}

phase4() {
    echo "=== Phase 4: Evaluation ==="
    if [ ! -f "$TRAIN_OUT" ] || [ ! -f "$TEST_OUT" ]; then
        echo "ERROR: Dataset not found. Run --phase2 first."
        exit 1
    fi
    cd "$PHASE4_DIR"
    if command -v jupyter &> /dev/null; then
        jupyter notebook evaluation.ipynb
    else
        echo "Jupyter not found. Open manually:"
        echo "  cd ${PHASE4_DIR} && jupyter notebook evaluation.ipynb"
    fi
    cd "$BASE_DIR"
}

if [ $# -eq 0 ]; then usage; fi

case "$1" in
    --phase1) phase1 ;;
    --phase2) phase2 ;;
    --phase3) phase3 ;;
    --phase4) phase4 ;;
    --all)
        phase2
        echo ""
        read -p "Run Phase 3 (training)? This may take hours [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            phase3
        fi
        phase4
        ;;
    *) usage ;;
esac
