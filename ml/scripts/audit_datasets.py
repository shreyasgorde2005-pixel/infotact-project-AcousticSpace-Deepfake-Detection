import os
import csv

DATASETS = {
    "rirs": "../data/rirs",
    "mismatch_dataset": "../data/mismatch_dataset",
    "tts_samples": "../data/tts_samples",
    "deepfake_mismatch_dataset": "../data/deepfake_mismatch_dataset",
    "demo_test_set": "../data/demo_test_set",
    "attack_test_set": "../data/attack_test_set",
    "defense_training_set": "../data/defense_training_set",
    "generalization_test_set": "../data/generalization_test_set",
    "demo_samples": "../data/demo_samples",
    "demo_showcase": "../data/demo_showcase",
}

def resolve_folder(folder):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rel_script = os.path.abspath(os.path.join(script_dir, folder))
    if os.path.isdir(rel_script):
        return rel_script
    if os.path.isdir(folder):
        return folder
    alt = folder.replace("../data", "data")
    if os.path.isdir(alt):
        return alt
    return folder

def count_wavs(folder):
    target = resolve_folder(folder)
    if not os.path.isdir(target):
        return 0
    return len([f for f in os.listdir(target) if f.endswith(".wav")])

def check_labels_csv(folder):
    target = resolve_folder(folder)
    path = os.path.join(target, "labels.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        return len(rows)

if __name__ == "__main__":
    print(f"{'Dataset':<28}{'WAV files':<12}{'labels.csv rows':<18}{'Status'}")
    for name, folder in DATASETS.items():
        n_wav = count_wavs(folder)
        n_csv = check_labels_csv(folder)
        if n_csv is None:
            status = "n/a (no labels.csv)"
        elif n_wav == n_csv:
            status = "OK"
        else:
            status = "MISMATCH - check!"
        print(f"{name:<28}{n_wav:<12}{str(n_csv):<18}{status}")
