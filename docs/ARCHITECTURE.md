# AcousticSpace System Architecture & Data Flow

This document outlines the end-to-end architecture of **AcousticSpace**, a room impulse response (RIR) mismatch detection system designed to identify synthetic and adversarial speech deepfakes by analyzing acoustic environment consistency.

---

## High-Level Architecture Diagram

```
+-------------------------------------------------------------------------------+
|                             Frontend (React + Vite)                           |
|                                                                               |
|   +-----------------------+     +-------------------+    +----------------+   |
|   |  AudioUpload Component| <-> |  WebSocket Hook   |    |  History Page  |   |
|   +-----------------------+     +-------------------+    +----------------+   |
+-------------------------------------------+-----------------------------------+
                                            |
                         HTTP POST /api/v1/predict  |  WebSocket /ws/predict
                         HTTP GET  /history         |  (Live Stage Streaming)
                                            v
+-------------------------------------------------------------------------------+
|                            Backend Service (FastAPI)                          |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |  Endpoint Routers (/api/v1/predict, /ws/predict, /history, /health)  |   |
|   +-----------------------------------------------------------------------+   |
|                                           |                                   |
|   +-----------------------+               v              +----------------+   |
|   |  Validation & CORS    | ->  +-------------------+ -> |  History Log   |   |
|   |  (Extension / Size)   |     |  Inference Engine |    |  (In-Memory)   |   |
|   +-----------------------+     +-------------------+    +----------------+   |
+-------------------------------------------+-----------------------------------+
                                            |
                                            v
+-------------------------------------------------------------------------------+
|                               ML Pipeline Layer                               |
|                                                                               |
|   +-----------------------+     +-------------------+    +----------------+   |
|   |   Audio Loading       | --> |  Feature Extract  | -> |  RIR Mismatch  |   |
|   |   (librosa / 16kHz)   |     |  (MFCC / Chroma / |    |  Classifier    |   |
|   |                       |     |   Spectral Cents) |    |  (PyTorch/AST) |   |
|   +-----------------------+     +-------------------+    +----------------+   |
+-------------------------------------------------------------------------------+
```

---

## Core System Components

### 1. Frontend Layer (`frontend/`)
- **Technology Stack:** React 18, TypeScript, Vite, TailwindCSS (Dark Mode UI).
- **Upload Modes:**
  - **HTTP POST Mode (`/api/v1/predict`):** Direct multipart audio upload returning the confidence score, prediction label (`fake` / `real`), and feature breakdown.
  - **WebSocket Live Mode (`/ws/predict`):** Establishes an interactive socket connection that streams granular stage-by-stage progress (`"Extracting acoustic features..."`, `"Evaluating RIR mismatch..."`, etc.) before returning the final inference payload.
- **Analysis History & Dashboard:** Queries `/history` to render past analyses and aggregate statistical metrics (total evaluations, deepfake detection percentage, average model confidence).

### 2. Backend Service (`backend/`)
- **Technology Stack:** Python 3.12, FastAPI, Uvicorn, Starlette WebSocket.
- **Input Validation:**
  - Strict file extension filtering (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
  - Payload size limiting (rejections > 20 MB with `400 Bad Request`).
  - Corrupted audio header detection via audio decoding fallback checks (`422 Unprocessable Entity`).
- **Prediction Logging (`app/core/history.py`):**
  - Maintains an ordered record of recent analyses with timestamps, file names, predictions, confidence percentages, and execution time.

### 3. ML Feature Extraction & Classification (`ml/`)
- **Acoustic Environment Verification:**
  - Standardizes input audio to `16,000 Hz` mono waveforms.
  - Extracts spectral features and evaluates environmental reverberation consistency.
- **Adversarial Robustness:**
  - Trained against RIR-matched synthetic audio and adversarial evasion attacks (`ml/data/attack_test_set/`), ensuring robust detection even when deepfakes simulate realistic room acoustics.

---

## Docker Containerization (`docker-compose.yml`)
The entire stack is containerized for seamless reproducibility during reviews:
- **Backend Service:** Bound to `http://localhost:8000` with automated health checks.
- **Frontend Development Server:** Accessible via `http://localhost:5173` with proxying configured to the FastAPI backend.
