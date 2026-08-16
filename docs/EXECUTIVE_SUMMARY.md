# AcousticSpace - Executive Summary
## Problem
Deepfake voices are increasingly hard to detect by ear alone. AcousticSpace
detects them by checking whether a voice's room echo matches its
background's room echo, a signal deepfake generators don't account for.
## Approach

We built our own labeled dataset simulating room-echo mismatch, fine-tuned
an Audio Spectrogram Transformer on it, and fused its score with an
independent breathing-pattern signal.
## Final Results (final_model_v1.pt)
| Test | Accuracy | EER |
|---|---|---|
| Held-out demo set | 0.800 | 0.250 |
| Attack test | 0.875 | 0.150 |
| Generalization test | 0.650 | 0.400 |
## Key Finding
We reproduced a real published attack (matching a deepfake's room echo to
its background) and found our baseline model was vulnerable to it. After
retraining with adversarial examples, accuracy on the attack set improved from 15.0% to 87.5%.
## What's Next
Real-world deployment would need testing against real (not synthetic)
recording conditions, a larger and more diverse dataset, and further
hardening against adversarial attacks beyond the one we tested.
