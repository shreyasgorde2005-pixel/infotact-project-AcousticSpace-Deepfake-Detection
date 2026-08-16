import torch
import librosa
import csv
import os
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from evaluate_utils import compute_eer, compute_accuracy

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHECKPOINT_PATH = "../checkpoints/best_model_defended.pt"
GEN_DIR = "../data/generalization_test_set"

def load_generalization_set():
    labels_csv = os.path.join(GEN_DIR, "labels.csv")
    file_paths, labels = [], []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_paths.append(os.path.join(GEN_DIR, row["filename"]))
            labels.append(int(row["label"]))
    return file_paths, labels

if __name__ == "__main__":
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID, num_labels=2, ignore_mismatched_sizes=True)
    
    # Use best_model.pt if best_model_defended.pt does not exist
    if not os.path.exists(CHECKPOINT_PATH):
        CHECKPOINT_PATH = "../checkpoints/best_model.pt"
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    
    file_paths, labels = load_generalization_set()
    all_scores, all_preds = [], []
    
    for path, true_label in zip(file_paths, labels):
        waveform, _ = librosa.load(path, sr=16000)
        inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            spoof_score = probs[1].item()
            pred = int(spoof_score > 0.5)
            
        all_scores.append(spoof_score)
        all_preds.append(pred)
        
        status = "CORRECT" if pred == true_label else "WRONG"
        print(f" {os.path.basename(path)}: true={true_label}, pred={pred}, score={spoof_score:.3f} [{status}]")
        
    acc = compute_accuracy(labels, all_preds)
    eer, _ = compute_eer(labels, all_scores)
    print(f"\nGeneralization test: accuracy={acc:.3f}, EER={eer:.3f}")
