from __future__ import annotations

import json
from pathlib import Path

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vocal_restoration.audio_io import load_audio, trim_silence


def waveform_metrics(signal: np.ndarray, sr: int):
    signal = np.asarray(signal, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak = float(np.max(np.abs(signal)))
    energy = float(np.mean(np.square(signal)))
    duration = float(len(signal) / sr)
    return {"rms": rms, "peak": peak, "energy": energy, "duration": duration}


def compute_stft(signal: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 256):
    stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    return stft, db


def estimate_pitch(signal: np.ndarray, sr: int):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        signal, fmin=50, fmax=1000, sr=sr, frame_length=2048
    )
    valid = voiced_flag
    if not np.any(valid):
        return {"mean_f0": 0.0, "median_f0": 0.0, "std_f0": 0.0, "voiced_fraction": 0.0}

    values = f0[valid]
    return {
        "mean_f0": float(np.nanmean(values)),
        "median_f0": float(np.median(values)),
        "std_f0": float(np.nanstd(values)),
        "voiced_fraction": float(np.mean(valid)),
    }


def estimate_formants(signal: np.ndarray, sr: int, order: int = 12):
    if len(signal) < order + 5:
        return {"f1": 0.0, "f2": 0.0, "f3": 0.0, "f4": 0.0}

    lpc = librosa.lpc(signal, order=order)
    roots = np.roots(lpc)
    roots = roots[np.imag(roots) >= 0]
    freqs = np.angle(roots) * sr / (2 * np.pi)
    freqs = freqs[(freqs > 80) & (freqs < sr / 2)]
    freqs = np.sort(freqs)

    if len(freqs) == 0:
        return {"f1": 0.0, "f2": 0.0, "f3": 0.0, "f4": 0.0}

    formants = np.asarray(freqs[:4], dtype=float)
    while len(formants) < 4:
        formants = np.append(formants, 0.0)

    return {"f1": float(formants[0]), "f2": float(formants[1]), "f3": float(formants[2]), "f4": float(formants[3])}


def compute_cross_correlation(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compare_audio_pair(original: np.ndarray, processed: np.ndarray, sr: int):
    orig = trim_silence(np.asarray(original, dtype=np.float32))
    proc = trim_silence(np.asarray(processed, dtype=np.float32))
    n = min(len(orig), len(proc))
    orig = orig[:n]
    proc = proc[:n]

    orig_spec, orig_db = compute_stft(orig, sr)
    proc_spec, proc_db = compute_stft(proc, sr)

    mean_abs_diff = float(np.mean(np.abs(orig - proc)))
    spectral_dev = float(np.mean(np.abs(orig_db - proc_db)))
    correlation = compute_cross_correlation(orig, proc)
    rms_ratio = float(np.sqrt(np.mean(np.square(proc))) / (np.sqrt(np.mean(np.square(orig))) + 1e-8))

    return {
        "mean_abs_diff": mean_abs_diff,
        "spectral_deviation_db": spectral_dev,
        "cross_correlation": correlation,
        "rms_ratio": rms_ratio,
        "pitch_original": estimate_pitch(orig, sr),
        "pitch_processed": estimate_pitch(proc, sr),
        "formants_original": estimate_formants(orig, sr),
        "formants_processed": estimate_formants(proc, sr),
    }


def save_waveform_plot(output_path: Path, original: np.ndarray, processed: np.ndarray, sr: int):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(original)) / sr
    axes[0].plot(t, original, color="C0", alpha=0.9, linewidth=1.0)
    axes[0].set_title("Original waveform")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t[: min(len(original), len(processed))], processed[: min(len(original), len(processed))], color="C1", alpha=0.8, linewidth=1.0)
    axes[1].set_title("Processed waveform")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_spectrogram_plot(output_path: Path, original: np.ndarray, processed: np.ndarray, sr: int):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for idx, signal in enumerate([original, processed]):
        spectrogram = np.abs(librosa.stft(signal, n_fft=2048, hop_length=256))
        db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="linear", ax=axes[idx])
        axes[idx].set_title(["Original spectrogram", "Processed spectrogram"][idx])

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def analyze_signal_relationships(
    original_path: str | Path,
    isolated_path: str | Path,
    residual_path: str | Path | None = None,
    recombined_path: str | Path | None = None,
    output_dir: str | Path = "results",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original, sr = load_audio(original_path, target_sr=22050)
    isolated, _ = load_audio(isolated_path, target_sr=sr)

    report = {
        "paths": {
            "original": str(original_path),
            "isolated": str(isolated_path),
            "residual": str(residual_path) if residual_path else None,
            "recombined": str(recombined_path) if recombined_path else None,
        },
        "original_metrics": waveform_metrics(original, sr),
        "isolated_metrics": waveform_metrics(isolated, sr),
    }

    orig_vs_iso = compare_audio_pair(original, isolated, sr)
    report["original_vs_isolated"] = orig_vs_iso

    if residual_path is not None:
        residual, _ = load_audio(residual_path, target_sr=sr)
        residual_metrics = waveform_metrics(residual, sr)
        report["residual_metrics"] = residual_metrics
        report["isolated_vs_residual"] = compare_audio_pair(isolated, residual, sr)

    if recombined_path is not None:
        recombined, _ = load_audio(recombined_path, target_sr=sr)
        report["recombined_metrics"] = waveform_metrics(recombined, sr)
        report["original_vs_recombined"] = compare_audio_pair(original, recombined, sr)

    save_waveform_plot(output_dir / "waveform_comparison.png", original, isolated, sr)
    save_spectrogram_plot(output_dir / "spectrogram_comparison.png", original, isolated, sr)

    with open(output_dir / "analysis_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return report
