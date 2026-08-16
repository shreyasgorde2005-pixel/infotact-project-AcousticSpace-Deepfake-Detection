import os
MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "../ml/checkpoints/best_model.pt")
MODEL_VERSION_NAME = os.getenv("MODEL_VERSION_NAME", "baseline")
DEVICE = os.getenv("DEVICE", "cpu")
