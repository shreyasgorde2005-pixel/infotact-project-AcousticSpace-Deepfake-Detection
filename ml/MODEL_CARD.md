# AcousticSpace - Model Card

## Base architecture
Audio Spectrogram Transformer (AST), MIT/ast-finetuned-audioset-10-10-0.4593,
fine-tuned for binary classification (matched vs. mismatched audio).

## Training data
Combined dataset: RIR-mismatch pairs, TTS-deepfake-mismatch pairs, and
RIR-matched adversarial defense examples. See DATA_CARD.md for full counts.

## Fusion
The final prediction combines the AST's RIR-mismatch score (70% weight)
with a breathing-pattern naturalness heuristic (30% weight) - see
ml/scripts/fusion.py.

## Final reported metrics
| Test set | Accuracy | EER |
|---|---|---|
| Held-out demo set | 0.800 | 0.250 |
| Attack test (after defense) | 0.875 | 0.150 |
| Generalization test | 0.650 | 0.400 |

## Known limitations
- Trained on a small, fully synthetic dataset - real-world performance is
untested.
- The breathing-naturalness score is a hand-written heuristic, not a
trained classifier.
- Explainability (`flagged_segments`) uses windowed scoring, not true
saliency or Grad-CAM.

## Checkpoint
`checkpoints/final_model_v1.pt` - frozen on Day 16 for the final review.
Do not overwrite this file with further experiments; save any new
checkpoints under a different name if you keep iterating afterward.
