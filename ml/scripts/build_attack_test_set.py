import numpy as np
import soundfile as sf
import librosa
import os
import csv
import glob
from scipy.signal import fftconvolve
 
RIR_DIR = "../data/rirs"
TTS_DIR = "../data/tts_samples"
OUT_DIR = "../data/attack_test_set"
SR = 16000
 
def load_rir(name):
    rir, _ = librosa.load(f"{RIR_DIR}/{name}.wav", sr=SR)
    return rir / np.max(np.abs(rir))
 
def convolve_and_normalize(signal, rir):
    out = fftconvolve(signal, rir)[: len(signal)]
    return out / (np.max(np.abs(out)) + 1e-8)
 
def generate_background_noise(length, rir, noise_level=0.05):
    noise = np.random.normal(0, 1, length)
    noise = convolve_and_normalize(noise, rir)
    return noise * noise_level
 
def build_attack_sample(tts_path, room, out_name):
    # Convolve the fake voice with the SAME room as its background, so voice
    # and background now acoustically match despite the voice still being fake.
    voice, _ = librosa.load(tts_path, sr=SR)
    rir = load_rir(room)
    voice_conv = convolve_and_normalize(voice, rir)
    bg = generate_background_noise(len(voice), rir)
    mix = voice_conv + bg
    sf.write(f"{OUT_DIR}/{out_name}.wav", mix, SR)
 
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rooms = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(f"{RIR_DIR}/*.wav"))]
    tts_files = sorted(glob.glob(f"{TTS_DIR}/*.wav"))
 
    if not tts_files:
        print("No TTS files found in data/tts_samples/ - run synthesize_tts.py first.")
    else:
        rows = []
        for i, tts_path in enumerate(tts_files):
            room = rooms[i % len(rooms)]
            name = f"attack_{i:03d}"
            build_attack_sample(tts_path, room, name)
            # true label stays 1 (still a deepfake) - the point is testing whether
            # matching the RIR fools the model into predicting 0
            rows.append((f"{name}.wav", 1, room))
 
        with open(f"{OUT_DIR}/labels.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "room"])
            writer.writerows(rows)
 
        print(f"Generated {len(rows)} RIR-matched adversarial deepfake samples in {OUT_DIR}/")
        print("All true labels are 1 (deepfake) - testing whether RIR-matching fools the detector.")
