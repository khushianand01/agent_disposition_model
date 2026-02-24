#!/bin/bash
# run_best_training.sh
echo "Starting Production Training - Best Data Version (V6 Balanced)..."

# Activate Env
source /home/ubuntu/disposition_model/venv/bin/activate

# Stop any running training safely
pkill -f train_production.py

# Final Production Paths
export TRAIN_PATH="/home/ubuntu/disposition_model/data/production/train_best.json"
export VAL_PATH="/home/ubuntu/disposition_model/data/production/val_best.json"
export OUTPUT_DIR="/home/ubuntu/disposition_model/outputs/qwen_3b_production_best"

# Ensure we are in the correct directory
cd /home/ubuntu/disposition_model/qwen_3b


# Run Training (Optimized for Tesla T4 / Unsloth)
nohup python3 train_production.py \
    --data_path "$TRAIN_PATH" \
    --val_data_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --save_steps 500 \
    --eval_steps 500 \
    --load_best_model_at_end True > training_production_resume.log 2>&1 &

echo "Training started in background."
echo "Monitor progress: tail -f /home/ubuntu/disposition_model/qwen_3b/training_production_resume.log"
