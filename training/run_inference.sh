#!/bin/bash
# run_inference.sh

echo "Starting Inference CLI..."
/home/ubuntu/disposition_model/venv/bin/python3 qwen_3b/inference/inference.py --model_path /home/ubuntu/disposition_model/qwen_3b/outputs/qwen_3b_production_best
