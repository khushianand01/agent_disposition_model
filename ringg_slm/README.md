# Ringg SLM Directory

This directory contains all Ringg-1.5B-specific code and scripts.

## Structure

- **inference/** - Inference and evaluation scripts
  - `eval_v11_baseline.py` - Baseline evaluation script
  
- **data_prep/** - Data preparation scripts (Stage-2 will be added here)

- **train_ringg_slm.py** - Training script using Unsloth
- **test_ringg_slm.py** - Testing script

## Current Status

- **Stage 1 Training**: IN PROGRESS (PID 44002, ETA ~5 hours)
  - Training on `train_v11_s1.json` (18,835 samples)
  - Validation on `val_v11_s1.json` (1,032 samples)
  - Output: `outputs/ringg_slm_stage1`

- **Stage 2**: Full extraction - TBD
