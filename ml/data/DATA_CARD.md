# AcousticSpace - Dataset Card

## Final counts (Week 3 wrap-up)
- Synthetic RIRs: 25
- RIR-mismatch dataset: 250
- TTS deepfake samples: 5
- Deepfake-mismatch dataset: 10
- Demo test set (held out): 5
- Attack test set: 5
- Defense training set: 125
- Generalization test set: 10

## Synthetic RIRs
25 room configurations (small_room, medium_room, large_hall, bathroom, office, plus 20 randomized room configurations room_00 to room_19) generated with pyroomacoustics using the Sabine equation, targeting RT60 values between 0.2s and 1.2s.

## Mismatch dataset
For each voice sample and each room, we generate:
- a MATCHED pair: voice + background noise both convolved with the same RIR (label 0)
- a MISMATCHED pair: voice convolved with one room's RIR, background noise convolved with a different room's RIR (label 1)

Total samples: 250 (125 matched / 125 mismatched pairs)
Voice samples used: 5

## TTS & Deepfake Mismatch Dataset
- TTS deepfake samples: 5 synthetic speech clips generated via pyttsx3.
- Deepfake-mismatch dataset: 10 samples combining real and TTS speech convolved with RIRs and acoustic backgrounds.

## Test Sets & Training Data
- Demo test set (held out): 5 samples curated via prepare_demo_set.py (data/demo_test_set/), used for the team's end-to-end integration test - not used during training.
- Attack test set: 5 adversarial RIR-matched deepfake samples (data/attack_test_set/) to evaluate model robustness against RIR-matching attacks.
- Defense training set: 125 RIR-matched adversarial training samples (data/defense_training_set/) to train the model against RIR-matching spoofing attacks.

## Generalization test set
Built from TTS phrases and room configurations never used anywhere in training or the earlier held-out/attack test sets. This is a synthetic distribution-shift test, not a real external corpus (e.g. MLAAD or an "in-the-wild" dataset) - a real external corpus is a good next upgrade if time and dataset access allow.
## Training & Test Sets (Adversarial & Generalization)
- Demo test set (held out): 5 samples curated via prepare_demo_set.py (data/demo_test_set/), used for the team's end-to-end integration test - not used during training.
- Attack test set: 5 adversarial RIR-matched deepfake samples (data/attack_test_set/) to evaluate model robustness against RIR-matching attacks.
- Defense training set: 125 RIR-matched adversarial training samples (data/defense_training_set/) to train the model against RIR-matching spoofing attacks.
- Generalization test set: 10 samples (data/generalization_test_set/) representing unseen noise and room conditions to evaluate out-of-distribution performance.

## Known limitations
- Background "noise" is currently synthetic white/pink noise shaped by a convolved RIR, not a real recorded ambience track - a real limitation to mention honestly in the report.
- All RIRs are simulated (shoebox rooms), not measured in physical spaces.
