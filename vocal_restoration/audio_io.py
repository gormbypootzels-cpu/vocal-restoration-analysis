from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int | None = None):
    """Load audio and convert to mono float32 if needed."""
    path = str(path)
    y, sr = sf.read(path, dtype="float32", always_2d=False)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if target_sr is not None and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y.astype(np.float32), sr


def save_audio(path: str | Path, signal: np.ndarray, sr: int):
    """Save mono float audio to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(signal, dtype=np.float32), sr)


def trim_silence(signal: np.ndarray, threshold: float = 0.01, frame_length: int = 2048):
    """Trim leading and trailing silence."""
    if signal.size == 0:
        return signal

    non_silent = np.abs(signal) > threshold
    if not np.any(non_silent):
        return signal[:1]

    start = np.argmax(non_silent)
    end = len(signal) - np.argmax(non_silent[::-1])
    return signal[start:end]
