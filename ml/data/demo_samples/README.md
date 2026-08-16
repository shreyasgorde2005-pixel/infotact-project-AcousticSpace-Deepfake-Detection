# Demo Samples

| File | What it is | Expected result |
| --- | --- | --- |
| `clear_real.wav` | Genuine speech, matched RIR (`sample1_pair14_matched_room_10.wav`) | Misclassified as fake (`is_fake=True`, `rir_mismatch_score=0.752`, `confidence=0.676`) |
| `clear_fake.wav` | TTS deepfake, mismatched RIR (`sample4_pair6_mismatched_room_02_vs_room_03.wav`) | Flagged as fake (`is_fake=True`, `rir_mismatch_score=0.696`, `confidence=0.637`) |
| `edge_case.wav` | RIR-matched adversarial deepfake (`attack_000.wav`) | Flagged as fake (`is_fake=True`, `rir_mismatch_score=0.817`, `confidence=0.722`) - honest attack moment to present live |
| `backup_alternate.wav` | Spare, in case another file misbehaves (`sample1_pair3_matched_office.wav`) | Spare audio clip (`is_fake=True`, `rir_mismatch_score=0.710`, `confidence=0.647`) |

## Inferred Baseline Metrics (via `scripts/inference.py`)

- **`clear_real.wav`**: `is_fake=True`, `confidence=0.676`, `rir_mismatch_score=0.752`, `breathing_score=0.500`
- **`clear_fake.wav`**: `is_fake=True`, `confidence=0.637`, `rir_mismatch_score=0.696`, `breathing_score=0.500`
- **`edge_case.wav`**: `is_fake=True`, `confidence=0.722`, `rir_mismatch_score=0.817`, `breathing_score=0.500`
- **`backup_alternate.wav`**: `is_fake=True`, `confidence=0.647`, `rir_mismatch_score=0.710`, `breathing_score=0.500`
