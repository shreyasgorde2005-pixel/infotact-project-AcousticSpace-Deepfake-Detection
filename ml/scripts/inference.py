import torch
import librosa
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import os
from fusion import get_breathing_score, fuse_scores
from explainability import find_suspicious_segments

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt"),
)

_feature_extractor = None
_model = None

def load_model():
    global _feature_extractor, _model
    if _model is None:
        _feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
        _model = ASTForAudioClassification.from_pretrained(
            MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
        )
        if os.path.exists(CHECKPOINT_PATH):
            _model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
            print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        else:
            print("No checkpoint found - using base pretrained weights (predictions will be unreliable).")
        _model.eval()
    return _feature_extractor, _model

def predict_mismatch(audio_path: str, sr: int = 16000, fast_mode: bool = False) -> dict:
    feature_extractor, model = load_model()
    waveform, _ = librosa.load(audio_path, sr=sr)
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        spoof_score = probs[1].item()
        breathing_score = get_breathing_score(audio_path, sr)
        fused_score = fuse_scores(spoof_score, breathing_score)
        flagged_segments = find_suspicious_segments(audio_path, model, feature_extractor, sr=sr, fast_mode=fast_mode)
    return {
        "is_fake": fused_score > 0.5,
        "confidence": round(max(fused_score, 1 - fused_score), 3),
        "rir_mismatch_score": round(spoof_score, 3),
        "breathing_score": round(breathing_score, 3),
        "flagged_segments": flagged_segments,
    }

if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "../data/samples/sample1.wav"
    result = predict_mismatch(test_path)
    print(result)
