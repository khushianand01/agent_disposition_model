#!/bin/bash
echo "Starting v6 Training (All Dispositions Balanced)..."

# Activate Env
source /home/ubuntu/miniconda3/bin/activate disposition_v3

# Stop any running training safely
pkill -f train_production.py

# Set Vars
export TRAIN_PATH="data/splits/train_v6.json"
export VAL_PATH="data/splits/val_v6.json"
export OUTPUT_DIR="outputs/qwen3_8b_v6_balanced"

# Run Training
nohup python3 train_production.py \
    --data_path $TRAIN_PATH \
    --val_data_path $VAL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --load_best_model_at_end True > training_v6.log 2>&1 &

echo "Training started in background with nohup. Monitor training_v5.log"

echo "Training Complete. Model saved to $OUTPUT_DIR"
