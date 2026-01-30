# Qwen 3B Model Directory

This directory contains all Qwen-3B-specific code and scripts.

## Structure

- **inference/** - Inference scripts for Qwen model
  - `inference.py` - Main inference class with prompt formatting and date/amount validation
  - `evaluate_final.py` - Evaluation scripts
  
- **deployment/** - API deployment scripts
  - `app.py` - FastAPI server for Qwen inference

- **data_prep/** - Data preparation scripts (Stage-2 will be added here)

## Current Status

- **Stage 1**: Classification only (disposition + payment_disposition)
- **Stage 2**: Full extraction (all 7 fields) - TBD
