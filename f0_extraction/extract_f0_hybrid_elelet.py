#!/usr/bin/env python3
"""Extract automatic H1 and DDSP F0 contours with the hybrid Elelet tracker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tf_animalysis_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "tf_animalysis_cache"))

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f0_extraction.hybrid_elelet import HybridEleletConfig, make_elelet, track_hybrid_elelet


def _load_audio(path: Path, sr: int) -> np.ndarray:
    audio, file_sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio


def parse_args() -> argparse.Namespace:
    defaults = HybridEleletConfig()
    parser = argparse.ArgumentParser(
        description="Track noisy elephant rumbles with an Elelet-ridge + Elelet-SHRP Viterbi path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Folder containing WAV files.")
    parser.add_argument("--output", type=Path, default=None, help="Output folder (default: INPUT/f0_hybrid_elelet).")
    parser.add_argument("--sr", type=int, default=defaults.sr)
    parser.add_argument("--hop-length", type=int, default=defaults.hop_length)
    parser.add_argument("--kernel-size", type=int, default=defaults.kernel_size)
    parser.add_argument("--supp-mult", type=float, default=defaults.supp_mult)
    parser.add_argument("--num-channels", type=int, default=defaults.num_channels)
    parser.add_argument("--transform-fmin", type=float, default=defaults.transform_fmin)
    parser.add_argument("--transform-fmax", type=float, default=defaults.transform_fmax)
    parser.add_argument("--h1-min", type=float, default=defaults.h1_min)
    parser.add_argument("--h1-max", type=float, default=defaults.h1_max)
    parser.add_argument("--candidate-step-hz", type=float, default=defaults.candidate_step_hz)
    parser.add_argument("--n-harmonics", type=int, default=defaults.n_harmonics)
    parser.add_argument("--harmonic-bin-radius", type=int, default=defaults.harmonic_bin_radius)
    parser.add_argument("--ridge-weight", type=float, default=defaults.ridge_weight)
    parser.add_argument("--interharmonic-penalty", type=float, default=defaults.interharmonic_penalty)
    parser.add_argument("--refinement-radius-hz", type=float, default=defaults.refinement_radius_hz)
    parser.add_argument("--smoothness", type=float, default=defaults.smoothness)
    parser.add_argument("--max-jump-hz", type=float, default=defaults.max_jump_hz)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files (smoke tests).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (args.input / "f0_hybrid_elelet")
    output.mkdir(parents=True, exist_ok=True)
    config = HybridEleletConfig(
        sr=args.sr,
        hop_length=args.hop_length,
        kernel_size=args.kernel_size,
        supp_mult=args.supp_mult,
        num_channels=args.num_channels,
        transform_fmin=args.transform_fmin,
        transform_fmax=args.transform_fmax,
        h1_min=args.h1_min,
        h1_max=args.h1_max,
        candidate_step_hz=args.candidate_step_hz,
        n_harmonics=args.n_harmonics,
        harmonic_bin_radius=args.harmonic_bin_radius,
        ridge_weight=args.ridge_weight,
        interharmonic_penalty=args.interharmonic_penalty,
        refinement_radius_hz=args.refinement_radius_hz,
        smoothness=args.smoothness,
        max_jump_hz=args.max_jump_hz,
    )
    files = sorted(args.input.glob("*.wav"))
    if args.limit is not None:
        files = files[: args.limit]
    transform = make_elelet(config)

    failures: list[str] = []
    for path in tqdm(files, desc="Hybrid Elelet F0"):
        try:
            result = track_hybrid_elelet(_load_audio(path, config.sr), config, transform=transform)
            frame = pd.DataFrame(
                {
                    "time": result.time,
                    # Compatibility with the existing annotation/refinement pipeline.
                    "frequency": result.h1_hz,
                    # Explicit synthesis control: no ambiguous divide-by-two downstream.
                    "f0_hz": result.f0_hz,
                    "confidence": result.confidence,
                    "proposal_confidence": result.proposal_confidence,
                    "frequency_role": "f1",
                    "algorithm": "elelet_peak_shrp_power_viterbi",
                    "selected_h1_min_hz": result.selected_band_hz[0],
                    "selected_h1_max_hz": result.selected_band_hz[1],
                }
            )
            frame.to_csv(output / f"{path.stem}.f0.csv", index=False, float_format="%.6f")
        except Exception as exc:  # keep a dataset run alive and report all failures
            failures.append(f"{path.name}: {exc}")

    metadata = {"algorithm": "elelet_peak_shrp_power_viterbi", "parameters": vars(config), "failures": failures}
    (output / "metainfo.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(files) - len(failures)}/{len(files)} contours to {output}")
    if failures:
        print("Failures:")
        print("\n".join(f"  {item}" for item in failures))


if __name__ == "__main__":
    main()
