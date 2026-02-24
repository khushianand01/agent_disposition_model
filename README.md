# 📞 Disposition Extraction Model v7 (Production)

A high-performance AI pipeline for extracting structured **Call Dispositions**, **Payment Details**, and **Confidence Scores** from conversational transcripts using **Qwen-2.5-7B-Instruct**.

![Status](https://img.shields.io/badge/Status-Production-success)
![Model](https://img.shields.io/badge/Model-Qwen%202.5%207B%20(4--bit)-blue)
![Stack](https://img.shields.io/badge/Stack-Unsloth%20%7C%20FastAPI-blueviolet)

---

## 🚀 Key Improvements (v7)
| Feature | Old Model | **New Model (v7)** |
| :--- | :--- | :--- |
| **Base Model** | Qwen-2.5-3B | **Qwen-2.5-7B-Instruct** |
| **Model ID** | khushianand01/disposition_model_v6 | **khushianand01/disposition_model** |
| **Date Logic** | Guesswork | **Calendar-Aware** (Handles Feb-end, Parso, Kal) |
| **Stability** | Occasional Crashes | **100% JSON Validity** (Hard-Truncation Logic) |
| **Accuracy** | ~55% | **>85% (F1-score on key labels)** |

---

## 📁 Project Structure

```
disposition_model/
├── api/                            # 🚀 Production Service
│   ├── app.py                      #    FastAPI Server (Port 8005)
│   ├── inference.py                #    Unsloth Inference Engine
│   └── static/index.html           #    Web UI
├── data/production/                # 📊 Datasets
│   ├── train_best.json, val_best.json, test_best.json
├── docs/                           # 📑 Reports & Notes
│   ├── production_eval_report.txt  #    Round 8 Evaluation Results
│   ├── evaluation_results_audit.json #  Full Prediction Audit Trail
│   ├── training_missed.txt         #    Booster Phase Label Gaps
│   └── API_COMMANDS.md
├── scripts/                        # 🔧 Utilities
│   ├── evaluate.py                 #    Production Evaluation Script
│   └── test_transcripts.txt        #    Manual Test Cases
├── training/                       # 🏋️ Fine-tuning
│   ├── train_production.py
│   └── deployment/monitoring/      #    Prometheus/Grafana
├── preprocess/                     # 🔄 Data Prep Scripts
├── logs/                           # 📝 API & Eval Logs
└── requirements.txt
```

---

## 🏃‍♂️ Quick Start & Operations

### API Service (Managed by systemd)
```bash
# Start
sudo systemctl start disposition_api

# Stop
sudo systemctl stop disposition_api

# Restart (after code changes)
sudo systemctl restart disposition_api

# Check status
sudo systemctl status disposition_api
```

### Health Check
```bash
curl -s http://localhost:8005/health
```

### Quick Predict Test
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Main parso 5000 pay kar dunga.", "current_date": "2026-02-21"}' \
  | python3 -m json.tool
```

### Run Evaluation
```bash
python3 scripts/evaluate.py --samples 200
```

### 📈 Monitoring (All running at startup)
| Service | URL |
| :--- | :--- |
| **Grafana Dashboard** | http://localhost:3000 |
| **Prometheus** | http://localhost:9090 |
| **API Metrics** | http://localhost:8005/metrics |

```bash
# Check monitoring status
sudo systemctl status prometheus grafana-server

# Restart GPU Exporter if needed
python3 training/deployment/monitoring/gpu_exporter.py &
```

---

## 🛠️ Operational Improvements
- **"Parso" Recovery**: Automatically identifies Hinglish relative dates and converts to `current_date + 2`.
- **Job Mapping**: Explicitly maps `JOB_LOSS` to `JOB_CHANGED_WAITING_FOR_SALARY` for business logic alignment.
- **Calendar Check**: Uses `calendar.monthrange` to prevent hallucinations like "30th February".
- **Confidence Scoring**: Returns a `confidence_score` (0.0 - 1.0) for every extraction.

---

## 📧 Documentation & Support
- **Main Report**: [production_eval_report.txt](./production_eval_report.txt)
- **Monitoring**: See `training/deployment/monitoring/README_MONITORING.md`
- **Maintainer**: Khushi Anand
- **Version**: 7.0.0 (Production Ready)
