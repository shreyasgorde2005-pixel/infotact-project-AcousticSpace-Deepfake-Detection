import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATASETS = {
    "RIRs": "../data/rirs",
    "Mismatch\ndataset": "../data/mismatch_dataset",
    "TTS\nsamples": "../data/tts_samples",
    "Deepfake\nmismatch": "../data/deepfake_mismatch_dataset",
    "Demo\ntest set": "../data/demo_test_set",
    "Attack\ntest set": "../data/attack_test_set",
    "Defense\ntraining": "../data/defense_training_set",
    "Generalization\ntest set": "../data/generalization_test_set",
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

counts = {name: count_wavs(folder) for name, folder in DATASETS.items()}

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
bars = ax.bar(counts.keys(), counts.values(), color="#3B6FD1", edgecolor="#1E3A6E")
ax.set_ylabel("Number of audio files")
ax.set_title("AcousticSpace - Dataset Composition")

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        str(int(height)),
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

plt.xticks(fontsize=8)
plt.tight_layout()

# Save image to ml/data/dataset_composition.png
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.abspath(os.path.join(script_dir, "../data/dataset_composition.png"))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200)

print(f"saved dataset_composition.png to {out_path}")
