# 📞 Disposition Extraction Model v2

An end-to-end AI pipeline for extracting structured **Call Dispositions** and **Payment Details** from conversational transcripts using **Qwen/Qwen3-8B** (via Unsloth).

![Status](https://img.shields.io/badge/Status-Active-success)
![Model](https://img.shields.io/badge/Model-Qwen3%208B-blue)
![Stack](https://img.shields.io/badge/Stack-Unsloth%20%7C%20FastAPI%20%7C%20Docker-blueviolet)

## 🚀 Features
- **Fine-Tuned SLM**: Customized **Qwen/Qwen3-8B** model (Small Language Model) optimized for call center analytics.
- **Structured JSON Output**: Extracts 7 key fields including `disposition`, `ptp_amount`, `ptp_date`, etc.
- **Confidence Scoring**: Returns a confidence score (0-1) for every prediction to route low-confidence calls for manual review.
- **Production Ready**: 
    - **FastAPI** Service for real-time inference.
    - **Docker** Containerized for easy deployment.
    - **Unsloth** Optimization (2x faster training, 60% less memory).

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/khushianand01/Disposition_model2.git
cd Disposition_model2
```

### 2. Install Dependencies
```bash
# Recommended: Use a virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🏃‍♂️ Usage

### 1. Data Preparation (From Scratch)
To prepare the dataset with **Class Balancing** and **Synthetic Augmentation**:

```bash
# 1. OPTIONAL: Generate additional synthetic data for rare classes
python3 preprocess/generate_synthetic_extra.py

# 2. REQUIRED: Balance dataset (Merge Real + Synthetic -> Train/Val/Test Splits)
# This creates data/splits/train_final.json (used for training)
python3 preprocess/force_balance_dataset.py
```

### 2. Training (Fine-Tuning)
Run the training script (supports `nohup` for overnight runs):
```bash
# Start Training (on GPU)
python3 train/train_unsloth.py
```
*   **Output**: Saved to `outputs/qwen3_8b_lora_mapped`
*   **Logs**: Check `training.log` to monitor progress.

### 3. Evaluation
Run the full evaluation report on the test set:
```bash
python3 inference/evaluate_final.py
```
This generates:
- **Macro F1 Scores** per field.
- **Confidence Calibration** report.
- **Accuracy breakdown** for rare classes.

### 4. API Serving (FastAPI)
Start the REST API server:
```bash
python3 deployment/app.py
```
*   **Swagger UI**: Visit `http://localhost:8000/docs` to test interactively.

### 5. Docker Deployment
Build and run the containerized service:
```bash
# Build
docker build -t disposition-model .

# Run (Requires GPU)
docker run --gpus all -p 8000:8000 disposition-model
```

---

## 📊 Project Structure

```text
├── data/
│   ├── splits/             # train_final.json, val_final.json, test_final.json
│   └── processed/          # Intermediate files
│
├── deployment/
│   └── app.py              # FastAPI Service
│
├── inference/
│   ├── inference.py        # Core formatting & prediction logic
│   └── evaluate_final.py   # Metrics & Calibration calculation
│
├── preprocess/
│   ├── force_balance_dataset.py # MAIN script to create final datasets
│   ├── generate_synthetic_extra.py
│   └── clean_and_format.py
│
├── train/
│   └── train_unsloth.py    # Fine-tuning script (QLoRA)
│
├── outputs/                # Trained model artifacts
├── Dockerfile              # Container definition
└── README.md               # Project documentation
```

## 🔍 Model Details

### Input Format
```text
Agent: Hello, calling from Collections...
Borrower: I can pay 5000 next Tuesday.
```

### Output JSON
```json
{
  "disposition": "PROMISE_TO_PAY",
  "payment_disposition": "PTP",
  "reason_for_not_paying": null,
  "ptp_amount": "5000",
  "ptp_date": "next Tuesday",
  "followup_date": "next Tuesday",
  "remarks": "Borrower promised to pay 5000.",
  "confidence_score": 0.9852
}
```

---

## 📈 Performance
- **Training Time**: ~12 Hours on T4 GPU (1 Epoch)
- **Inference Speed**: ~5-8 seconds per call (T4 GPU)
- **Optimization**: Unsloth 4-bit quantization reduces VRAM usage by ~50%.

---

## 📧 Contact
Project Maintainer: Khushi Anand
