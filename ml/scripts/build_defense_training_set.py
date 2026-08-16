import numpy as np
import soundfile as sf
import librosa
import os
import csv
import glob
from scipy.signal import fftconvolve

RIR_DIR = "data/rirs" if os.path.exists("data/rirs") else "../data/rirs"
TTS_DIR = "data/tts_samples" if os.path.exists("data/tts_samples") else "../data/tts_samples"
OUT_DIR = "data/defense_training_set" if os.path.exists("data") else "../data/defense_training_set"
SR = 16000

def load_rir(name):
    rir, _ = librosa.load(f"{RIR_DIR}/{name}.wav", sr=SR)
    return rir / np.max(np.abs(rir))

def convolve_and_normalize(signal, rir):
    out = fftconvolve(signal, rir)[:len(signal)]
    return out / (np.max(np.abs(out)) + 1e-8)

def generate_background_noise(length, rir, noise_level=0.05):
    noise = np.random.normal(0, 1, length)
    noise = convolve_and_normalize(noise, rir)
    return noise * noise_level

def build_defense_sample(tts_path, room, out_name):
    # Same idea as the Day 9 attack set: voice and background convolved with
    # the SAME room, so the RIR mismatch signal disappears even though the
    # voice is still fake. This time it goes into training, not just testing.
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
        print("No TTS files found - run synthesize_tts.py (Day 7) first.")
    else:
        rows = []
        count = 0
        for tts_path in tts_files:
            for room in rooms:
                name = f"defense_{count:04d}"
                build_defense_sample(tts_path, room, name)
                rows.append((f"{name}.wav", 1, room))
                count += 1

        with open(f"{OUT_DIR}/labels.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "room"])
            writer.writerows(rows)

        print(f"Generated {len(rows)} RIR-matched adversarial training samples in {OUT_DIR}/")
        print("These get added to training so the model learns to catch this trick, not just be tested on it.")
