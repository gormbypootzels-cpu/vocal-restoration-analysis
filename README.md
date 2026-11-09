# Vocal Restoration Analysis

This project is a Python-based analysis and restoration toolkit for comparing original recordings to vocal-isolated outputs, residual noise, recombined mixtures, and restoration candidates.

It is designed to answer five questions:

1. What changed between the original and vocal-isolated recordings?
2. Which changes are responsible for the distorted or shifted vocal characteristics?
3. Which changes can be mathematically reversed or compensated for?
4. Can the residual and original signals recover vocal characteristics lost during isolation?
5. Which restoration configuration produces the closest measurable approximation of the original vocal signal?

## Design goals

- Preserve the original recording as the reference signal.
- Analyze isolated dialogue and residual noise independently.
- Compare original, isolated, residual, recombined, and restored signals on the same timeline.
- Quantify waveform, spectral, and pitch/formant differences.
- Generate multiple restoration candidates without inventing uncertain speech.
- Rank candidates by measurable fidelity and artifact severity.

## Recommended workflow

1. Load the original recording and the isolated dialogue output.
2. Load the residual signal and any recombined mixture.
3. Run the comparative analysis to compute signal metrics.
4. Inspect playback and plots for waveform and spectrogram changes.
5. Generate restoration candidates using conservative spectral-envelope and residual-aware corrections.
6. Score every candidate against the original and isolate the most faithful configuration.

## Repository layout

- `vocal_restoration/analysis.py` — waveform, spectrogram, F0, and formant analysis
- `vocal_restoration/restoration.py` — restoration candidate generation and ranking
- `vocal_restoration/cli.py` — command-line runner
- `vocal_restoration/audio_io.py` — audio loading and saving helpers

## Quick start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the analysis on your recordings:

```bash
python -m vocal_restoration.cli \
  --original path/to/original.wav \
  --isolated path/to/isolated.wav \
  --residual path/to/residual.wav \
  --recombined path/to/recombined.wav \
  --output-dir results
```

This produces:

- a JSON analysis report
- waveform plots
- spectrogram plots
- restoration candidate scores
- saved candidate audio files

## Important note

This project is intentionally conservative. Where the source signal is missing or uncertain, it distinguishes recoverable information from inferred reconstruction rather than silently inventing speech content.

## License

MIT
