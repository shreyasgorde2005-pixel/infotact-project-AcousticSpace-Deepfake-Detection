import sys
sys.path.append(".")
from breathing_detector import breathing_naturalness_score
import librosa
import numpy as np

def fuse_scores(rir_mismatch_score, breathing_score, rir_weight=0.7, breathing_weight=0.3):
    # Combines the AST model's RIR-mismatch score with the breathing
    # naturalness score into one fused confidence (higher = more likely fake).
    # breathing_score is "naturalness" (higher = more natural/real), so we
    # invert it before combining.
    breathing_fake_signal = 1 - breathing_score
    fused = (rir_weight * rir_mismatch_score) + (breathing_weight * breathing_fake_signal)
    return float(np.clip(fused, 0, 1))

def get_breathing_score(audio_path, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    return breathing_naturalness_score(y, sr)

if __name__ == "__main__":
    import glob
    import os
    for path in sorted(glob.glob("../data/demo_test_set/*.wav")):
        breathing = get_breathing_score(path)
        fake_rir_score = 0.6  # placeholder for standalone testing
        fused = fuse_scores(fake_rir_score, breathing)
        print(f"{os.path.basename(path)}: breathing={breathing:.2f}, fused_score={fused:.2f}")
