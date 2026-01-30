#!/bin/bash
echo "Starting Ringg SLM Stage-1 Training..."

# Activate Env
source /home/ubuntu/miniconda3/bin/activate disposition_v3

# Stop any running Ringg training safely
pkill -f train_ringg_slm.py

# Run Training
export CUDA_VISIBLE_DEVICES=0
python3 ringg_slm/train_ringg_slm.py \
    --num_train_epochs 3 \
    --output_dir ringg_slm/outputs/ringg_slm_stage1 > ringg_stage1.log 2>&1 &

echo "Training started in background. Monitor ringg_stage1.log"
