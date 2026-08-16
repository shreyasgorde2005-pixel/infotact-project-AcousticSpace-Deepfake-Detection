# 3. System Architecture

The system architecture consists of several interconnected layers designed for scalability, performance, and clear separation of concerns:

- **Frontend (React)**: The user interface where users upload audio files and view predictions. Built with React for a responsive and interactive experience.
- **Backend (FastAPI, Dockerized)**: The core API server built with FastAPI. It handles incoming requests, orchestrates the data pipeline, and communicates with the ML model. The entire backend is containerized using Docker to ensure consistent environments.
- **Data Pipeline**: Responsible for preprocessing incoming audio files (e.g., feature extraction, normalization) to prepare them for the ML model.
- **ML Model (AST + fusion)**: The core deepfake detection model using an Audio Spectrogram Transformer (AST) with feature fusion techniques to analyze and classify the audio accurately.

## Key Endpoints

The backend exposes several key endpoints to interact with the system:
- `/predict`: Endpoint to submit audio files for deepfake detection.
- `/history`: Retrieves a history of past predictions and results.
- `/ws/predict`: A WebSocket endpoint for streaming predictions or receiving real-time updates.
- `/model-info`: Provides metadata and details about the currently loaded ML model.
