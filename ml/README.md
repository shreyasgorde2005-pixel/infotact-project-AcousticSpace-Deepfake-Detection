# AcousticSpace - ML & Data Pipeline

This folder contains everything related to dataset generation, model training, and evaluation for AcousticSpace.

## Start here
- `data/DATA_CARD.md` - what each of the 8 datasets is and how big it is
- `data/REPRODUCIBILITY.md` - how to regenerate everything from scratch
- `MODEL_CARD.md` - the frozen model's architecture, training data, and final metrics
- `../docs/results_log.md` - every test result, final numbers marked clearly

## Folder structure
- `scripts/` - all generation, training, and evaluation scripts
- `data/` - generated datasets (gitignored - regenerate using data/README.md)
- `checkpoints/` - trained model weights (gitignored - regenerate using train_baseline.py)

## Quickstart
See `data/README.md` for the full 9-step dataset regeneration order, then run `scripts/train_baseline.py` to train, and `scripts/final_headline_results.py` to reproduce the final reported numbers.
