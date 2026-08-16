from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import PredictionResponse, SegmentFlag, AcousticComparison
from app.core.history import log_prediction, get_history
import sys
import os
import tempfile
import time
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "scripts"))
from inference import predict_mismatch
from segment_features import extract_voice_bg_features

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB = 20

app = FastAPI(title="AcousticSpace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.core.config import MODEL_VERSION_NAME, MODEL_CHECKPOINT_PATH

@app.get("/")
def root():
    return {
        "name": "AcousticSpace API",
        "version": "1.0",
        "model_version": MODEL_VERSION_NAME,
        "docs": "/docs",
    }

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {str(exc)}"},
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {"model_version": MODEL_VERSION_NAME, "checkpoint_path": MODEL_CHECKPOINT_PATH}

TIMEOUT_SECONDS = float(os.getenv("PREDICT_TIMEOUT_SECONDS", "30"))

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_v1(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit)",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    start_time = time.time()
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, predict_mismatch, tmp_path),
            timeout=TIMEOUT_SECONDS,
        )
        acoustic_feats = extract_voice_bg_features(tmp_path)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Prediction timed out after {TIMEOUT_SECONDS}s")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process audio file: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    inference_time_ms = round((time.time() - start_time) * 1000, 1)
    log_prediction(file.filename, result["is_fake"], result["confidence"])
    return PredictionResponse(
        filename=file.filename,
        is_fake=result["is_fake"],
        confidence=result["confidence"],
        rir_mismatch_score=result["rir_mismatch_score"],
        breathing_score=result["breathing_score"],
        flagged_segments=[SegmentFlag(**s) for s in result["flagged_segments"]],
        acoustic_comparison=AcousticComparison(**acoustic_feats) if "acoustic_feats" in locals() else None,
        inference_time_ms=inference_time_ms,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_legacy(file: UploadFile = File(...)):
    return await predict_v1(file)

@app.get("/history")
def history():
    return {"history": get_history()}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        contents = await websocket.receive_bytes()
        await websocket.send_json({"stage": "received", "message": "File received"})
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
            
        await websocket.send_json({"stage": "extracting", "message": "Extracting audio features"})
        try:
            result = predict_mismatch(tmp_path)
        except Exception as e:
            await websocket.send_json({"stage": "error", "message": str(e)})
            return
        finally:
            os.remove(tmp_path)
            
        await websocket.send_json({"stage": "running_model", "message": "Running AST classifier"})
        log_prediction("live_stream.wav", result["is_fake"], result["confidence"])
        await websocket.send_json({
            "stage": "done",
            "result": {
                "is_fake": result["is_fake"],
                "confidence": result["confidence"],
                "rir_mismatch_score": result["rir_mismatch_score"],
                "breathing_score": result["breathing_score"],
                "flagged_segments": result["flagged_segments"],
            },
        })
    except WebSocketDisconnect:
        print("Client disconnected")
