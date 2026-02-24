from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import threading
import subprocess
import time
import pandas as pd
import io
import time

# Add project root and qwen_3b to sys.path so we can import inference module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.inference import get_model

app = FastAPI(title="Disposition Extraction API", version="1.0")
Instrumentator().instrument(app).expose(app)

# Prometheus custom metrics
REQUEST_COUNT = Counter("disposition_requests_total", "Total number of /predict requests")
REQUEST_ERRORS = Counter("disposition_request_errors_total", "Total number of failed /predict requests")
INFERENCE_TIME = Histogram("disposition_inference_seconds", "Inference latency in seconds")
MODEL_LOADED = Gauge("disposition_model_loaded", "Whether the model is loaded (1 = loaded)")
GPU_AVAILABLE = Gauge("disposition_gpu_available", "Whether CUDA GPU is available (1/0)")
# Per-GPU metrics will be labeled by index
GPU_UTIL = Gauge("disposition_gpu_util_percent", "GPU utilization percent", ["gpu"])
GPU_MEM_TOTAL = Gauge("disposition_gpu_mem_total_mb", "GPU memory total (MB)", ["gpu"])
GPU_MEM_USED = Gauge("disposition_gpu_mem_used_mb", "GPU memory used (MB)", ["gpu"])

def collect_gpu_metrics(period_s: int = 5):
    """Background thread: poll nvidia-smi and update Prometheus gauges."""
    while True:
        try:
            # Query utilization, memory total and used for all GPUs
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            GPU_AVAILABLE.set(1)
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    idx, util, mem_total, mem_used = parts[0], parts[1], parts[2], parts[3]
                    GPU_UTIL.labels(gpu=idx).set(float(util))
                    GPU_MEM_TOTAL.labels(gpu=idx).set(float(mem_total))
                    GPU_MEM_USED.labels(gpu=idx).set(float(mem_used))
        except Exception:
            # Nvidia-smi not available or failed
            GPU_AVAILABLE.set(0)
        time.sleep(period_s)

# Start GPU collection thread
gpu_thread = threading.Thread(target=collect_gpu_metrics, args=(5,), daemon=True)
gpu_thread.start()

# Serve static UI files (place `index.html` and logo under `qwen_3b/deployment/static`)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

from datetime import date

# Request Model
class TranscriptRequest(BaseModel):
    transcript: str
    current_date: str | None = None

# Nested Model for PTP Details
class PtpDetails(BaseModel):
    amount: float | None = None
    date: str | None = None

# Response Model
class DispositionResponse(BaseModel):
    disposition: str | None = None
    payment_disposition: str | None = None
    reason_for_not_paying: str | None = None
    ptp_details: PtpDetails | None = None
    remarks: str | None = None
    confidence_score: float | None = None

print("Loading model for API...")
# Initialize model on startup
model = get_model()

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "Qwen/Qwen2.5-7B-Instruct-LoRA"}

@app.get("/")
def read_root():
    # Serve the simple UI
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "running", "message": "Disposition Extraction API is active. Use /predict for inference or /docs for documentation."}


@app.post("/upload")
async def upload_and_process(file: UploadFile = File(...), output_format: str = Form("csv")):
    """Accepts CSV/Excel/JSON file with a transcript column, runs model.predict on each row, and returns a downloadable file."""
    filename = file.filename or f"upload_{int(time.time())}"
    body = await file.read()
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(body))
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(body))
        elif filename.lower().endswith('.json'):
            df = pd.read_json(io.BytesIO(body))
        else:
            # Try CSV by default
            df = pd.read_csv(io.BytesIO(body))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {e}")

    # Find transcript column
    transcript_col = None
    for c in df.columns:
        if 'transcript' in str(c).lower() or 'text' in str(c).lower() or 'conversation' in str(c).lower():
            transcript_col = c
            break
    if transcript_col is None:
        raise HTTPException(status_code=400, detail="No transcript/text column found in uploaded file.")

    results = []
    for _, row in df.iterrows():
        try:
            transcript = str(row.get(transcript_col, '') or '')
            pred = model.predict(transcript)
            # Flatten prediction into a dict row
            if isinstance(pred, dict):
                out = pred.copy()
            else:
                out = {"raw": str(pred)}
        except Exception as e:
            out = {"error": str(e)}
        # Keep original columns if needed
        out["_original_transcript"] = transcript
        results.append(out)

    out_df = pd.DataFrame(results)

    buf = io.BytesIO()
    output_filename = f"predictions_{int(time.time())}.{output_format}"
    if output_format == 'csv':
        out_df.to_csv(buf, index=False)
        buf.seek(0)
        media_type = 'text/csv'
    elif output_format in ('xlsx', 'xls'):
        out_writer = pd.ExcelWriter(buf, engine='openpyxl')
        out_df.to_excel(out_writer, index=False)
        out_writer.save()
        buf.seek(0)
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif output_format == 'json':
        buf.write(out_df.to_json(orient='records').encode())
        buf.seek(0)
        media_type = 'application/json'
    else:
        raise HTTPException(status_code=400, detail="Unsupported output format")

    return StreamingResponse(buf, media_type=media_type, headers={"Content-Disposition": f"attachment; filename={output_filename}"})

@app.post("/predict", response_model=DispositionResponse)
def predict_disposition(request: TranscriptRequest):
    REQUEST_COUNT.inc()
    if not request.transcript.strip():
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=400, detail="Transcript is empty")

    pred_date = request.current_date or str(date.today())
    start_t = time.time()
    try:
        with INFERENCE_TIME.time():
            result = model.predict(request.transcript, current_date=pred_date)

        if isinstance(result, dict) and "error" in result:
            REQUEST_ERRORS.inc()
            raise HTTPException(status_code=500, detail="Model failed to generate valid JSON")

        return result
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
