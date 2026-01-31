#!/bin/bash
echo "Starting Ringg SLM Stage-1 Training..."

# Activate Env
source /home/ubuntu/miniconda3/bin/activate disposition_v3

# Stop any running Ringg training safely
pkill -f train_ringg_slm.py

# Run Training
export CUDA_VISIBLE_DEVICES=0
python3 ringg_slm/train_ringg_slm.py \
    --data_path data/splits/train_v11_s1.json \
    --val_data_path data/splits/val_v11_s1.json \
    --num_train_epochs 3 \
    --output_dir outputs/ringg_slm_stage1 \
    --resume_from_checkpoint outputs/ringg_slm_stage1/checkpoint-1500 > ringg_stage1.log 2>&1 &

echo "Training started in background. Monitor ringg_stage1.log"
