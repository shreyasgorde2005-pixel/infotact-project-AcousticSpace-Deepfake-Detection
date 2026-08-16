import librosa
import numpy as np

def detect_pauses(waveform, sr, frame_length=1024, hop_length=256,
                  energy_threshold_db=-40, min_pause_duration=0.15):
    # Detects likely pause/breath segments using short-time energy.
    # Returns a list of (start_time, end_time) tuples.
    energy = np.array([
        np.sum(np.abs(waveform[i:i+frame_length])**2)
        for i in range(0, len(waveform), hop_length)
    ])
    energy_db = 10 * np.log10(energy / np.max(energy) + 1e-12)
    is_pause = energy_db < energy_threshold_db
    pauses = []
    start_idx = None
    for i, val in enumerate(is_pause):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            duration = (i - start_idx) * hop_length / sr
            if duration >= min_pause_duration:
                pauses.append((start_idx * hop_length / sr, i * hop_length / sr))
            start_idx = None
    return pauses

def breathing_naturalness_score(waveform, sr):
    # Real speech tends to have pauses spaced roughly every 1.5-5 seconds
    # (natural breathing rhythm). This is a simplification - a production
    # system would use a trained classifier instead of a hand-written rule.
    pauses = detect_pauses(waveform, sr)
    if len(pauses) < 2:
        return 0.5  # not enough data to judge, neutral score
    gaps = np.array([pauses[i+1][0] - pauses[i][1] for i in range(len(pauses) - 1)])
    natural_mask = (gaps >= 1.5) & (gaps <= 5.0)
    naturalness = natural_mask.mean() if len(gaps) > 0 else 0.5
    return float(naturalness)

if __name__ == "__main__":
    import glob
    import os

    data_dir = "data/demo_test_set" if os.path.exists("data/demo_test_set") else "../data/demo_test_set"
    for path in sorted(glob.glob(f"{data_dir}/*.wav")):
        y, sr = librosa.load(path, sr=16000)
        pauses = detect_pauses(y, sr)
        score = breathing_naturalness_score(y, sr)
        print(f"{path}: {len(pauses)} pauses detected, naturalness_score={score:.2f}")
