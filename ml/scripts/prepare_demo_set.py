import csv
import os
import random
import shutil

SRC_DIR = "../data/mismatch_dataset"
DEMO_DIR = "../data/demo_test_set"

def main():
    os.makedirs(DEMO_DIR, exist_ok=True)
    with open(f"{SRC_DIR}/labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
        
    random.seed(42)
    sample = random.sample(rows, min(5, len(rows)))
    
    demo_rows = []
    for row in sample:
        src = os.path.join(SRC_DIR, row["filename"])
        dst = os.path.join(DEMO_DIR, row["filename"])
        shutil.copy(src, dst)
        demo_rows.append(row)
        
    with open(f"{DEMO_DIR}/labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "voice_room", "background_room"])
        writer.writeheader()
        writer.writerows(demo_rows)
        
    print(f"Copied {len(demo_rows)} demo samples into {DEMO_DIR}/")

if __name__ == "__main__":
    main()
