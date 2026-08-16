# How the Dataset Was Built

To train and test AcousticSpace, we needed audio where we controlled exactly which room echo applied to the voice and which applied to the background — something no existing public dataset provides directly. So we built our own dataset in three stages.

First, we simulated realistic room acoustics using `pyroomacoustics`, generating 20 rooms of varying size and reflectivity — a small bathroom, a large hall, everything in between — each producing a unique Room Impulse Response (RIR).

Second, we needed both genuine and deepfake voice content. For genuine samples, we used real recorded speech convolved with a single room's RIR for both voice and background, so they acoustically match, exactly like a real recording. For deepfake samples, we generated synthetic speech using text-to-speech (TTS), then convolved the voice with one room's RIR and the background with a different room's RIR, simulating what happens when a fake voice is spliced into a mismatched recording.

Third, once we discovered our own model was vulnerable to a published attack — matching the deepfake's RIR to its background to hide the mismatch — we generated an additional adversarial training set using that same technique, so the model could learn to catch it too.

In total, we built 8 distinct datasets across the project: room RIRs, matched/mismatched pairs, TTS deepfake speech, combined deepfake-mismatch data, a held-out test set, an attack test set, defense training data, and a generalization test set with entirely unseen phrases and rooms. Every one is fully reproducible from a documented script pipeline (see `ml/data/REPRODUCIBILITY.md`).
