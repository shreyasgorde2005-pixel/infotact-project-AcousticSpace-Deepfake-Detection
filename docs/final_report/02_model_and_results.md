# 2. Model & Results

## Architecture and training
The base architecture uses an Audio Spectrogram Transformer (AST), MIT/ast-finetuned-audioset-10-10-0.4593, which was fine-tuned for binary classification to distinguish matched versus mismatched audio. The training data consists of a combined dataset containing RIR-mismatch pairs, TTS-deepfake-mismatch pairs, and RIR-matched adversarial defense examples. The final prediction fuses the AST's RIR-mismatch score (70% weight) with a breathing-pattern naturalness heuristic (30% weight).

## Final results
| Test set | Accuracy | EER |
|---|---|---|
| Held-out demo set | 0.800 | 0.250 |
| Attack test set | 0.875 | 0.150 |
| Generalization test set | 0.650 | 0.400 |

## Key finding: the attack-vs-defense story
During the Day 9 attack test, the baseline model was shown to be highly vulnerable when deepfake speech was convolved with a matching room RIR, dropping its accuracy to a mere 15.0%. This demonstrated that the model was overly reliant on the RIR-mismatch signal, which could be easily masked. The Day 11 defense retraining addressed this vulnerability by training on an adversarial dataset, which drastically improved the model's attack-test accuracy to a robust 87.5%.

## Key finding: generalization
The generalization test from Day 14 revealed a performance drop of 15% in accuracy when evaluating the model on completely unseen phrases, speakers, and rooms. This is an expected generalization gap for a model trained on a relatively small and fully synthetic dataset, indicating it is somewhat brittle to novel acoustic environments.
