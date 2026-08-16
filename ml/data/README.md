# Regenerating the AcousticSpace dataset

Run these scripts from `ml/scripts/`, in this exact order:

1. `generate_rirs.py` - creates 20 randomized room RIRs
2. `build_mismatch_dataset.py` - builds matched/mismatched voice+background pairs
3. `synthesize_tts.py` - synthesizes TTS deepfake speech (needs espeak on Linux)
4. `build_deepfake_mismatch_dataset.py` - combines real+TTS speech with RIRs
5. `prepare_demo_set.py` - curates a held-out demo/test set
6. `build_attack_test_set.py` - builds the RIR-matched adversarial attack set
7. `segment_features.py` - RT60/DRR feature extraction (importable module)
8. `build_defense_training_set.py` - builds RIR-matched adversarial *training* data (Day 11)
9. `build_generalization_set.py` - builds the unseen-condition generalization set (Day 14)

Steps 1-2 must run before everything else. Step 3 must run before step 4.
Steps 8 and 9 only need steps 1 and 3 to have already run.
