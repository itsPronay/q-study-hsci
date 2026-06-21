#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Quanto PTQ sweep — weights only, no group_size (quanto doesn't use it)
# nbits: 2, 4, 8, 88 (88 = float8)
#
# Usage:
#   bash script/quanto_script.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODELS=("sf" "ssm" "mvit" "mf")
DATASETS=("UP" "NF" "HC" "Pavia" "Indian" "Houston")

NBITS=(2 4 8 88)   # 88 = float8

QUANT_METHOD="quanto"
WANDB_MODE="online"

echo "================================================================="
echo "  HSI PTQ Grid Search — Quanto"
echo "  Models:   ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  nbits:    ${NBITS[*]} (88=float8)"
echo "================================================================="

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do

        echo ""
        echo "================================================================="
        echo "  TRAIN  |  model=$MODEL  dataset=$DATASET"
        echo "================================================================="
        python QHSIC/train.py \
            --model   "$MODEL" \
            --dataset "$DATASET"

        echo ""
        echo "  Train complete → ./model/${MODEL}_${DATASET}.pt"
        echo "-----------------------------------------------------------------"

        for NBITS_VAL in "${NBITS[@]}"; do

            echo ""
            echo "  EVAL  |  model=$MODEL  dataset=$DATASET  nbits=$NBITS_VAL"
            echo "-----------------------------------------------------------------"
            python QHSIC/eval.py \
                --model        "$MODEL" \
                --dataset      "$DATASET" \
                --quant_method "$QUANT_METHOD" \
                --nbits        "$NBITS_VAL" \
                --wandb_mode   "$WANDB_MODE"

        done

        echo ""
        echo "  Finished all nbits for model=$MODEL dataset=$DATASET"
        echo "================================================================="
    done
done

echo ""
echo "================================================================="
echo "  Quanto grid search complete."
echo "================================================================="