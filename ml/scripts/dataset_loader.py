import csv
import os

def load_mismatch_dataset(dataset_dir="../data/mismatch_dataset"):
    """
    Returns (file_paths, labels).
    labels: 0 = matched (genuine-like), 1 = mismatched (spoof-like)
    """
    labels_csv = os.path.join(dataset_dir, "labels.csv")
    file_paths, labels = [], []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_paths.append(os.path.join(dataset_dir, row["filename"]))
            labels.append(int(row["label"]))
    return file_paths, labels

def load_deepfake_dataset(dataset_dir="../data/deepfake_mismatch_dataset"):
    labels_csv = os.path.join(dataset_dir, "labels.csv")
    file_paths, labels = [], []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_paths.append(os.path.join(dataset_dir, row["filename"]))
            labels.append(int(row["label"]))
    return file_paths, labels

def load_combined_dataset():
    p1, l1 = load_mismatch_dataset()
    try:
        p2, l2 = load_deepfake_dataset()
    except FileNotFoundError:
        print("No deepfake_mismatch_dataset found yet - training on RIR-mismatch data only.")
        p2, l2 = [], []
    return p1 + p2, l1 + l2

def load_defense_dataset(dataset_dir="../data/defense_training_set"):
    labels_csv = os.path.join(dataset_dir, "labels.csv")
    file_paths, labels = [], []

    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_paths.append(os.path.join(dataset_dir, row["filename"]))
            labels.append(int(row["label"]))
    return file_paths, labels

def load_all_training_data():
    p1, l1 = load_combined_dataset()
    try:
        p2, l2 = load_defense_dataset()
    except FileNotFoundError:
        print("No defense_training_set found yet - training without adversarial examples.")
        p2, l2 = [], []
    return p1 + p2, l1 + l2


if __name__ == "__main__":
    paths, labels = load_mismatch_dataset()
    n_mismatched = sum(labels)
    print(f"Loaded {len(paths)} samples - {n_mismatched} mismatched, {len(labels) - n_mismatched} matched")
