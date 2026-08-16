import os
import sys
import glob
import librosa

def verify_audio_files(directory, expected_sr=16000):
    if not os.path.exists(directory):
        print(f"[SKIP] Directory not found: {directory}")
        return True

    wav_files = glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"[INFO] No .wav files found in {directory}")
        return True

    print(f"[CHECK] Verifying {len(wav_files)} files in {directory}...")
    errors = 0
    for file_path in wav_files:
        try:
            _, sr = librosa.load(file_path, sr=None, duration=0.1)
            if sr != expected_sr:
                print(f"  [WARN] Unexpected SR {sr} in {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {file_path}: {e}")
            errors += 1

    print(f"  -> Completed with {errors} errors.")
    return errors == 0

def main():
    root_data = os.path.join(os.path.dirname(__file__), "..", "data")
    dirs_to_check = [
        os.path.join(root_data, "rirs"),
        os.path.join(root_data, "mismatch_dataset"),
        os.path.join(root_data, "tts_samples"),
        os.path.join(root_data, "demo_test_set"),
    ]

    all_ok = True
    print("=== AcousticSpace Dataset Verification ===")
    for d in dirs_to_check:
        if not verify_audio_files(d):
            all_ok = False

    if all_ok:
        print("\n[SUCCESS] All dataset audio checks passed cleanly.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some audio files failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()
