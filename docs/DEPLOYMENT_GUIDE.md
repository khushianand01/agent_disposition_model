# 🚀 Disposition Extraction Model: Deployment & Corporate Usage Guide

## 1. Executive Summary
The **Disposition Extraction Model ** is an AI-powered engine designed to automate the analysis of collection call transcripts. It accurately extracts call dispositions, payment commitments (PTP), and reasons for non-payment, enabling real-time data entry and automated follow-up workflows.

**Key Capabilities:**
*   **PTP Extraction**: Identifies payment amounts and dates.
*   **Date Normalization**: Converts relative dates (e.g., "parso", "kal", "next Monday") into specific ISO dates.
*   **Reason Analysis**: Maps complex borrower explanations (e.g., job loss, medical issues) to standardized business categories.
*   **High Precision**: 99%+ accuracy on critical status labels like "Wrong Number" and "Foreclosure".

## 2. Business Value: AI vs. Manual Entry
Currently, agents/callers manually select dispositions. This model provides three critical upgrades:

*   **Zero Subjectivity**: Agents often select "Others" or "Late Response" when they are in a hurry. The AI objectively analyzes the transcript to find the *true* reason (e.g., hidden PTPs, Job Loss, or Foreclosure intent).
*   **Reduced Wrap-Up Time**: Automating the disposition entry can save 30-60 seconds per call in "After Call Work" (ACW), significantly increasing daily call capacity per agent.
*   **Audit & Transparency**: The model provides a `confidence_score` and `ptp_details` for every call. Managers can audit cases where the agent's manual entry differs from the actual verbal commitment in the transcript.
*   **Automated Actioning**: PTP dates and amounts extracted by AI can be fed directly into the payment reminder system without manual intervention.

---

## 3. System Architecture
*   **Base Model**: Qwen-2.5-7B-Instruct (Fine-tuned for Indian Collection Context).
*   **Backend**: Python 3.10 with **Unsloth** for high-efficiency 4-bit inference.
*   **API Framework**: FastAPI for low-latency request handling.
*   **Monitoring**: Prometheus (metrics) + Grafana (visual dashboard).
*   **Infrastructure**: NVIDIA Tesla T4 GPU (or better).

---

## 4. Quick Start for IT/DevOps

### **Service Management**
The API runs as a systemd service named `disposition_api`.

| Action | Command |
| :--- | :--- |
| **Start API** | `sudo systemctl start disposition_api` |
| **Stop API** | `sudo systemctl stop disposition_api` |
| **Restart API** | `sudo systemctl restart disposition_api` |
| **Check Status** | `sudo systemctl status disposition_api` |

### **Health Check**
Verify the service is live and model is loaded:
```bash
curl -s http://localhost:8005/health
```

### **Quick Prediction Test (Copy-Paste)**
Test a real transcript to see the AI output:
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Agent: 5000 pending hai kab pay kareinge? Borrower: Parso pakka kar dunge sir.",
    "current_date": "2026-02-23"
  }' | python3 -m json.tool
```

---

## 5. Technical Integration Specification

### **Primary Endpoint: `/predict`**
Used for real-time analysis of a single call transcript.

*   **URL**: `http://<server-ip>:8005/predict`
*   **Method**: `POST`
*   **Headers**: `Content-Type: application/json`

#### **Request Parameters**
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `transcript` | String | Yes | The full text of the call conversation. |
| `current_date` | Date (YYYY-MM-DD) | No | Reference date for relative terms like "parso". Defaults to current server date. |

#### **Response Schema**
```json
{
  "disposition": "String (e.g., ANSWERED, WRONG_NUMBER)",
  "payment_disposition": "String (e.g., PTP, WANT_FORECLOSURE, None)",
  "reason_for_not_paying": "String (e.g., FUNDS_ISSUE, MEDICAL_ISSUE - added in v8)",
  "ptp_details": {
    "amount": "Number/String (The promised payment amount)",
    "date": "Date (Normalized payment date in YYYY-MM-DD)"
  },
  "remarks": "String (Short summary justifying the result)",
  "confidence_score": "Float (0.0 to 1.0 accuracy estimate)"
}
```

---

### **Batch Endpoint: `/upload`**
Used for processing historical logs or bulk audits.

*   **URL**: `http://<server-ip>:8005/upload`
*   **Method**: `POST`
*   **Body**: `multipart/form-data`
    *   `file`: The CSV/Excel file containing a "transcript" column.
    *   `output_format`: "csv", "json", or "xlsx".

---

### **Developer Example (Python)**
Copy-paste this snippet for the integration team:

```python
import requests
import json

def get_disposition(transcript, server_url="http://localhost:8005"):
    payload = {
        "transcript": transcript,
        "current_date": "2026-02-23" # Optional: strictly fix the ref date
    }
    
    response = requests.post(f"{server_url}/predict", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Integration Call
result = get_disposition("Agent: Pending amount 5000. Borrower: Kal subah dunga.")
if result:
    print(f"Auto-Disposition: {result['disposition']}")
    print(f"PTP Date: {result['ptp_details']['date']}")
```

---

## 6. Performance Metrics
*   **Inference Latency**: ~4.9s per request (T4 GPU).
*   **Throughput**: ~840 transcripts/hour (batch mode).
*   **VRAM Footprint**: ~6.7 GB (Fixed).

---

## 7. Monitoring & Observability
Dashboards are available for real-time tracking of GPU usage, request volume, and model accuracy.
*   **Grafana Dashboard**: `http://<server-ip>:3000` (Login: admin/admin)
*   **Metrics Endpoint**: `http://<server-ip>:8005/metrics`

---

## 8. Roadmap & Optimization
*   **Booster v8**: Targeted retraining to push PTP recall from 62% to 85%+.
*   **Regional Support**: Migration to *Sarvam-2B* for Telugu/Tamil/Marathi dialect support.
*   **Latency Optimization**: Migration to *vLLM* for 2x concurrency improvement.

---
**Maintained by**: AI Engineering Team  
**Last Updated**: 2026-02-23
