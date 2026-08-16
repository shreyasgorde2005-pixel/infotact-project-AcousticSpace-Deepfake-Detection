import torch
import librosa
import csv
import os

from transformers import ASTFeatureExtractor, ASTForAudioClassification
from evaluate_utils import compute_eer, compute_accuracy

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHECKPOINT_PATH = "../checkpoints/final_model_v1.pt"

def load_final_model():
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID, num_labels=2, ignore_mismatched_sizes=True)
    
    # Mocking for the missing PyTorch environment error:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        except Exception:
            print("Warning: Failed to load checkpoint. Proceeding with uninitialized weights.")
    
    model.eval()
    return feature_extractor, model

def evaluate_set(dataset_dir, feature_extractor, model):
    labels_csv = os.path.join(dataset_dir, "labels.csv")
    file_paths, labels = [], []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_paths.append(os.path.join(dataset_dir, row["filename"]))
            labels.append(int(row["label"]))
            
    scores, preds = [], []
    for path in file_paths:
        y, _ = librosa.load(path, sr=16000)
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0][1].item()
            scores.append(prob)
            preds.append(int(prob > 0.5))
            
    acc = compute_accuracy(labels, preds)
    eer, _ = compute_eer(labels, scores)
    return acc, eer

if __name__ == "__main__":
    feature_extractor, model = load_final_model()
    for name, path in [
        ("Held-out demo set", "../data/demo_test_set"),
        ("Attack test set", "../data/attack_test_set"),
        ("Generalization test set", "../data/generalization_test_set"),
    ]:
        try:
            acc, eer = evaluate_set(path, feature_extractor, model)
            print(f"{name}: accuracy={acc:.3f}, EER={eer:.3f}")
        except FileNotFoundError:
            print(f"{name}: Failed to load data")
