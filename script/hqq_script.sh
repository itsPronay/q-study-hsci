#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Trains each model/dataset combo once, then sweeps all nbits × group_size.
#
# Usage:
#   bash script/hqq_script.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

  # Finished all bit/group combos for model=mvit dataset=HC , mf all baki
#
MODELS=(
    # "sf" 
    # "ssm" 
    "mvit" 
    # "mf"
    )
DATASETS=(
    # "UP" 
    # "NF" 
    # "HC" 
    "Pavia"  #sf pavia left, run later
    "Indian" 
    "Houston"
    )

NBITS=(
    1 
    2 
    3 
    4 
    8
    )
GROUP_SIZES=(
    8 
    16 
    32 
    64
    )

QUANT_METHOD="hqq"
WANDB_MODE="online"        # change to "disabled" to turn off W&B

echo "================================================================="
echo "  HSI PTQ Grid Search"
echo "  Models:      ${MODELS[*]}"
echo "  Datasets:    ${DATASETS[*]}"
echo "  nbits:       ${NBITS[*]}"
echo "  group_sizes: ${GROUP_SIZES[*]}"
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
        echo "  Train complete → ./model/${MODEL}.pt"
        echo "-----------------------------------------------------------------"

        for NBITS_VAL in "${NBITS[@]}"; do
            for GROUP_SIZE in "${GROUP_SIZES[@]}"; do

                echo ""
                echo "  EVAL  |  model=$MODEL  dataset=$DATASET  nbits=$NBITS_VAL  group_size=$GROUP_SIZE"
                echo "-----------------------------------------------------------------"
                python QHSIC/eval.py \
                    --model        "$MODEL" \
                    --dataset      "$DATASET" \
                    --quant_method "$QUANT_METHOD" \
                    --nbits        "$NBITS_VAL" \
                    --group_size   "$GROUP_SIZE" \
                    --wandb_mode   "$WANDB_MODE"

            done
        done

        echo ""
        echo "  Finished all bit/group combos for model=$MODEL dataset=$DATASET"
        echo "================================================================="
    done
done

echo ""
echo "================================================================="
echo "  Grid search complete."
echo "================================================================="