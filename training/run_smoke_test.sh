#!/bin/bash
# run_smoke_test.sh

echo "Running Sanity Check (Smoke Test)..."
/home/ubuntu/disposition_model/venv/bin/python3 qwen_3b/inference/smoke_test.py
