#!/bin/bash
# run_eval.sh

echo "Running Full Evaluation (2,388 samples)..."
echo "This will take ~60-80 minutes."
/home/ubuntu/disposition_model/venv/bin/python3 qwen_3b/inference/evaluate_final.py
