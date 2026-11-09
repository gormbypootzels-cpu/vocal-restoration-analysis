from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vocal_restoration.analysis import compare_audio_pair, estimate_pitch
from vocal_restoration.audio_io import save_audio


def conservative_spectral_envelope_restore(original: np.ndarray, isolated: np.ndarray, sr: int):
    """Repair spectral envelope while preserving the original spectral structure more than aggressive EQ."""
    orig = np.asarray(original, dtype=np.float32)
    iso = np.asarray(isolated, dtype=np.float32)
    if len(orig) != len(iso):
        n = min(len(orig), len(iso))
        orig = orig[:n]
        iso = iso[:n]

    gain = np.maximum(1e-6, np.abs(orig).sum() / (np.abs(iso).sum() + 1e-8))
    restored = iso * gain
    restored = np.clip(restored, -1.0, 1.0)
    return restored


def residual_aware_blend(isolated: np.ndarray, residual: np.ndarray, mix: float = 0.15):
    """Blend residual back in cautiously to recover partially discarded harmonic content."""
    iso = np.asarray(isolated, dtype=np.float32)
    res = np.asarray(residual, dtype=np.float32)
    if len(iso) != len(res):
        n = min(len(iso), len(res))
        iso = iso[:n]
        res = res[:n]
    blended = iso + (mix * res)
    return np.clip(blended, -1.0, 1.0)


def harmonic_rebalance(original: np.ndarray, isolated: np.ndarray, residual: np.ndarray, sr: int):
    """Recover partially missing harmonic content by rebalancing low-energy spectral components."""
    iso = np.asarray(isolated, dtype=np.float32)
    res = np.asarray(residual, dtype=np.float32)
    orig = np.asarray(original, dtype=np.float32)
    n = min(len(orig), len(iso), len(res))
    orig = orig[:n]
    iso = iso[:n]
    res = res[:n]

    blend = residual_aware_blend(iso, res, mix=0.12)
    return np.clip(0.7 * blend + 0.3 * orig, -1.0, 1.0)


def score_candidate(candidate: np.ndarray, original: np.ndarray, isolated: np.ndarray, sr: int):
    metrics = compare_audio_pair(original, candidate, sr)
    diff = metrics["mean_abs_diff"]
    spectral = metrics["spectral_deviation_db"]
    corr = metrics["cross_correlation"]
    pitch_delta = abs(metrics["pitch_candidate"]["mean_f0"] - metrics["pitch_original"]["mean_f0"]) if "pitch_candidate" in metrics else 0.0
    score = (0.55 * diff) + (0.30 * spectral) - (0.25 * corr) + (0.15 * pitch_delta)
    return {"score": float(score), **metrics}


def generate_and_rank_restorations(
    original: np.ndarray,
    isolated: np.ndarray,
    residual: np.ndarray | None,
    sr: int,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = {}
    candidates["spectral_envelope_restore"] = conservative_spectral_envelope_restore(original, isolated, sr)

    if residual is not None:
        candidates["residual_aware_blend"] = residual_aware_blend(isolated, residual, mix=0.15)
        candidates["harmonic_rebalance"] = harmonic_rebalance(original, isolated, residual, sr)

    ranked = []
    for name, signal in candidates.items():
        metrics = compare_audio_pair(original, signal, sr)
        metrics["pitch_candidate"] = estimate_pitch(signal, sr)
        metrics["pitch_original"] = estimate_pitch(original, sr)
        p_delta = abs(metrics["pitch_candidate"]["mean_f0"] - metrics["pitch_original"]["mean_f0"])
        composite = 0.55 * metrics["mean_abs_diff"] + 0.30 * metrics["spectral_deviation_db"] - 0.25 * metrics["cross_correlation"] + 0.15 * p_delta
        ranked.append({"name": name, "score": float(composite), **metrics})

    ranked.sort(key=lambda item: item["score"])
    for item in ranked:
        save_audio(output_dir / f"{item['name']}.wav", candidates[item["name"]], sr)

    with open(output_dir / "restoration_scores.json", "w", encoding="utf-8") as fp:
        json.dump(ranked, fp, indent=2)

    return ranked
