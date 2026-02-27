# 📞 Agent Disposition Model v7.2 (Production Ready)

A high-performance multilingual AI pipeline for extracting structured **Call Dispositions**, **Payment Intent**, **Reason for Not Paying**, and **Performance Entities (Date/Amount)** from conversational transcripts.

![Status](https://img.shields.io/badge/Status-Production-success)
![Model](https://img.shields.io/badge/Model-Qwen%202.5%207B%20(4--bit)-blue)
![Stack](https://img.shields.io/badge/Stack-Unsloth%20%7C%20FastAPI-blueviolet)

---

## 🛠️ Prerequisites & Hardware
To run this model with acceptable latency, the following is required:

*   **GPU:** Minimum **15GB VRAM** (e.g., NVIDIA Tesla T4, A10G, or A100).
*   **VRAM Usage:** ~14.1 GB (Loaded in 4-bit quantization).
*   **Memory:** 16GB+ RAM.
*   **Python:** 3.10+
*   **CUDA:** 12.1+ installed.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/khushianand01/agent_disposition_model.git
cd agent_disposition_model

# 2. Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the API

### Method 1: Web UI & Manual Run
```bash
venv/bin/python3 app.py
```
*   **Web Interface**: Access the interactive dashboard at `http://localhost:8005`
*   **API Docs**: Swagger UI available at `http://localhost:8005/docs`

### Method 2: Systemd Service (Recommended for Production)
```bash
# 1. Copy the service file
sudo cp disposition_api.service /etc/systemd/system/

# 2. Start the service
sudo systemctl daemon-reload
sudo systemctl enable disposition_api
sudo systemctl start disposition_api
```

---

## � Project Structure

```
agent_disposition_model/
├── api/                            # 🚀 Production Service
│   ├── app.py                      #    FastAPI Server (Port 8005)
│   ├── inference.py                #    Unsloth Inference Engine
│   └── static/index.html           #    Web UI
├── data/                           # 📊 Datasets (Optional)
├── docs/                           # 📑 Reports & Notes
├── eval_datasets/                  # 🧪 Test Sets
├── logs/                           # 📝 API & Eval Logs
├── requirements.txt                # Python dependencies
├── disposition_api.service         # Systemd unit
└── README.md                       # Project documentation
```

---

## 📊 Performance Benchmarks (Tesla T4)
| User Scenario | Response Time (Median) | Reliability |
| :--- | :--- | :--- |
| **1 Single User** | **~4.5 seconds** | 100% |
| **3 Concurrent Users** | ~12.5 seconds | 100% |
| **5 Concurrent Users** | ~22.2 seconds | 100% |

---

## 🌐 Intent Extraction Logic
- **Job Loss**: Maps intents to `JOB_CHANGED_WAITING_FOR_SALARY`.
- **Date Handling**: Resolves relative terms (*kal, parso*) into standard `YYYY-MM-DD` using the server's real-time clock.
- **Multilingual**: High accuracy across Hindi, Bengali, English, Marathi, Tamil, Telugu, and more.

---

## 📑 Maintainer & Support
- **Version**: 7.2 (Stable Handover)
- **Primary Repo**: [agent_disposition_model](https://github.com/khushianand01/agent_disposition_model)
- **Production Sync**: [disposition_websockets](https://github.com/khushianand308/disposition_websockets)
