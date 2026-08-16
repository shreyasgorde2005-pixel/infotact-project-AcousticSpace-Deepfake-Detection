# Reproducibility Appendix

## Environment
- Python 3.11
- Key packages: librosa, soundfile, pyroomacoustics, scipy, torch, transformers, scikit-learn, pyttsx3

## Full dataset generation sequence
See `data/README.md` for the exact 9-step order.

## Randomness and seeds
- `generate_rirs.py` uses `random.seed(7)` and `np.random.seed(7)`
- `build_generalization_set.py` uses `np.random.RandomState(99)`
- Other scripts do not fix a seed - regenerating them will produce slightly different audio each time. This is a known limitation worth disclosing rather than hiding.

## Disk space
Regenerating everything from scratch produces roughly 26 MB of audio files. All generated data is gitignored; only the generation scripts are version controlled.

## Known limitations (consolidated)
- Background noise is synthetic (shaped white/pink noise), not real recorded ambience.
- Room impulse responses are simulated shoebox rooms, not physically measured.
- The generalization test (Day 14) uses a synthetic distribution shift, not a real external corpus like MLAAD.
