import torch
import librosa
import numpy as np

def find_suspicious_segments(audio_path, model, feature_extractor, sr=16000, 
                             window_sec=1.0, hop_sec=0.5, threshold_margin=0.15,
                             fast_mode=False):
    if fast_mode:
        hop_sec = 1.0 # fewer windows scanned - faster, slightly coarser
    # Slides a window across the clip, scores each window independently, and
    # flags windows whose fake-probability is notably higher than the clip's
    # average. This approximates "where did the model focus" without needing
    # to hook into the transformer's internal attention weights.
    
    y, _ = librosa.load(audio_path, sr=sr)
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    
    scores = []
    times = []
    
    for start in range(0, max(1, len(y) - window_len), hop_len):
        segment = y[start:start + window_len]
        if len(segment) < window_len * 0.5:
            continue
            
        inputs = feature_extractor(segment, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0][1].item()
            
        scores.append(prob)
        times.append(start / sr)
        
    if not scores:
        return []
        
    avg_score = np.mean(scores)
    flagged = []
    
    for t, s in zip(times, scores):
        if s > avg_score + threshold_margin:
            flagged.append({
                "start_time": round(t, 2),
                "end_time": round(t + window_sec, 2),
                "reason": f"Elevated mismatch score ({s:.2f} vs avg {avg_score:.2f})",
            })
            
    return flagged
