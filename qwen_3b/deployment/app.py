from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from prometheus_fastapi_instrumentator import Instrumentator

# Add project root to sys.path so we can import inference module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.inference import get_model

app = FastAPI(title="Disposition Extraction API", version="1.0")
Instrumentator().instrument(app).expose(app)

from datetime import date

# Request Model
class TranscriptRequest(BaseModel):
    transcript: str
    current_date: str | None = None

# Response Model
class DispositionResponse(BaseModel):
    disposition: str | None = None
    payment_disposition: str | None = None
    reason_for_not_paying: str | None = None
    ptp_amount: float | str | None = None  # Updated to accommodate numeric extraction
    ptp_date: str | None = None
    followup_date: str | None = None
    remarks: str | None = None
    confidence_score: float | None = None

print("Loading model for API...")
# Initialize model on startup
model = get_model()

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "Qwen/Qwen3-8B-LoRA"}

@app.get("/")
def read_root():
    return {"status": "running", "message": "Disposition Extraction API is active. Use /predict for inference or /docs for documentation."}

@app.post("/predict", response_model=DispositionResponse)
def predict_disposition(request: TranscriptRequest):
    try:
        if not request.transcript.strip():
            raise HTTPException(status_code=400, detail="Transcript is empty")
        
        # Use provided date or default to server date (YYYY-MM-DD)
        pred_date = request.current_date or str(date.today())
            
        result = model.predict(request.transcript, current_date=pred_date)
        
        # Handle JSON parse error case from inference.py
        if isinstance(result, dict) and "error" in result:
             raise HTTPException(status_code=500, detail="Model failed to generate valid JSON")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
