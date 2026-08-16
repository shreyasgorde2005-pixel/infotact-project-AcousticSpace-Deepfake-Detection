import numpy as np
import soundfile as sf
import librosa
import os
import csv
import pyttsx3
from scipy.signal import fftconvolve

SR = 16000
OUT_DIR = "../data/generalization_test_set"

NEW_PHRASES = [
    "Good morning, this is an automated notification.",
    "Your package has been delivered to the front desk.",
    "We noticed unusual activity on your account.",
    "The train to the city center departs in ten minutes.",
    "Please hold while we transfer your call.",
]

def synthesize_new_phrases(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)  # deliberately different voice than Day 7
    engine.setProperty("rate", 165)
    paths = []
    for i, phrase in enumerate(NEW_PHRASES):
        path = f"{out_dir}/gen_tts_{i:02d}.wav"
        engine.save_to_file(phrase, path)
        paths.append(path)
    engine.runAndWait()
    return paths

def random_new_rooms(n=6, seed=99):
    import pyroomacoustics as pra
    rng = np.random.RandomState(seed)
    rooms = []
    for i in range(n):
        dim = [rng.uniform(3, 18), rng.uniform(2.5, 12), rng.uniform(2.3, 5)]
        rt60 = rng.uniform(0.15, 1.4)
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, dim)
            room = pra.ShoeBox(dim, fs=SR, materials=pra.Material(e_absorption), max_order=max_order)
            room.add_source([dim[0] / 2, dim[1] / 2, 1.5])
            room.add_microphone([dim[0] / 4, dim[1] / 4, 1.2])
            room.compute_rir()
            rir = room.rir[0][0]
            rir = rir / np.max(np.abs(rir))
            name = f"gen_room_{i:02d}"
            sf.write(f"{OUT_DIR}/{name}.wav", rir, SR)
            rooms.append(name)
        except Exception as e:
            print(f"Skipped room {i}: {e}")
    return rooms
import glob
from scipy.signal import fftconvolve

RIR_DIR = "../data/rirs" if os.path.exists("../data/rirs") else "data/rirs"
TTS_DIR = "../data/tts_samples" if os.path.exists("../data/tts_samples") else "data/tts_samples"
REAL_DIR = "../data/samples" if os.path.exists("../data/samples") else "data/samples"
OUT_DIR = "../data/generalization_test_set" if os.path.exists("../data") else "data/generalization_test_set"
SR = 16000

def load_rir(name):
    rir, _ = librosa.load(f"{RIR_DIR}/{name}.wav", sr=SR)
    return rir / np.max(np.abs(rir))

def convolve_and_normalize(signal, rir):
    out = fftconvolve(signal, rir)[:len(signal)]
    return out / (np.max(np.abs(out)) + 1e-8)
  
def generate_background_noise(length, rir, noise_level=0.1):
    noise = np.random.normal(0, 1, length)
    noise = convolve_and_normalize(noise, rir)
    return noise * noise_level

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    tts_paths = synthesize_new_phrases(OUT_DIR)
    rooms = random_new_rooms()
    rows = []
    for i, tts_path in enumerate(tts_paths):
        room_genuine = rooms[i % len(rooms)]
        room_mismatch = rooms[(i + 2) % len(rooms)]
        voice, _ = librosa.load(tts_path, sr=SR)
        rir_g, _ = librosa.load(f"{OUT_DIR}/{room_genuine}.wav", sr=SR)
        rir_g = rir_g / np.max(np.abs(rir_g))
        genuine_mix = convolve_and_normalize(voice, rir_g) + generate_background_noise(len(voice), rir_g)
        genuine_name = f"gen_genuine_{i:02d}.wav"
        sf.write(f"{OUT_DIR}/{genuine_name}", genuine_mix, SR)
        rows.append((genuine_name, 0))

        rir_m, _ = librosa.load(f"{OUT_DIR}/{room_mismatch}.wav", sr=SR)
        rir_m = rir_m / np.max(np.abs(rir_m))
        mismatch_mix = convolve_and_normalize(voice, rir_g) + generate_background_noise(len(voice), rir_m)
        mismatch_name = f"gen_mismatch_{i:02d}.wav"
        sf.write(f"{OUT_DIR}/{mismatch_name}", mismatch_mix, SR)
        rows.append((mismatch_name, 1))

    with open(f"{OUT_DIR}/labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} generalization-test samples using NEW phrases and NEW rooms in {OUT_DIR}/")
