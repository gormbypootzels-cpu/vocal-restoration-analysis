from __future__ import annotations

import argparse
from pathlib import Path

from vocal_restoration.analysis import analyze_signal_relationships
from vocal_restoration.audio_io import load_audio
from vocal_restoration.restoration import generate_and_rank_restorations


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze vocal isolation transformations and rank restoration candidates.")
    parser.add_argument("--original", required=True, help="Path to the original recording.")
    parser.add_argument("--isolated", required=True, help="Path to the vocal-isolated recording.")
    parser.add_argument("--residual", default=None, help="Optional path to the residual/noise signal.")
    parser.add_argument("--recombined", default=None, help="Optional path to the recombined signal.")
    parser.add_argument("--output-dir", default="results", help="Directory for analysis output.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original, sr = load_audio(args.original)
    isolated, _ = load_audio(args.isolated, target_sr=sr)
    residual = load_audio(args.residual, target_sr=sr)[0] if args.residual else None
    recombined = load_audio(args.recombined, target_sr=sr)[0] if args.recombined else None

    report = analyze_signal_relationships(
        original_path=args.original,
        isolated_path=args.isolated,
        residual_path=args.residual,
        recombined_path=args.recombined,
        output_dir=output_dir,
    )

    ranked = generate_and_rank_restorations(
        original=original,
        isolated=isolated,
        residual=residual,
        sr=sr,
        output_dir=output_dir / "restoration_candidates",
    )

    print("Analysis complete.")
    print(f"Saved report to: {output_dir / 'analysis_report.json'}")
    print(f"Saved restoration rankings to: {output_dir / 'restoration_candidates' / 'restoration_scores.json'}")
    if ranked:
        print("Top restoration candidate:")
        print(ranked[0])


if __name__ == "__main__":
    main()

