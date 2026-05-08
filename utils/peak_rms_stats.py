#!/usr/bin/env python3
"""
Compute RMS distribution after peak normalization for a dataset of WAV files.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf


def peak_normalize(audio):
    peak = np.max(np.abs(audio))
    if peak <= 0:
        return audio
    return audio / peak


def iter_wavs(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                yield Path(dirpath) / name


def compute_rms(audio, eps=1e-12):
    return float(np.sqrt(np.mean(audio ** 2) + eps))


def summarize(values):
    if not values:
        return {}
    arr = np.array(values, dtype=np.float32)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory with wav files")
    parser.add_argument("--max_files", type=int, default=0, help="Limit number of files (0 = all)")
    args = parser.parse_args()

    rms_values = []
    file_count = 0

    for wav_path in iter_wavs(args.root):
        try:
            audio, _ = sf.read(wav_path, dtype="float32")
        except Exception:
            continue

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = peak_normalize(audio)
        rms_values.append(compute_rms(audio))
        file_count += 1

        if args.max_files and file_count >= args.max_files:
            break

    stats = summarize(rms_values)
    if not stats:
        print("No wav files found.")
        return

    print(f"root: {args.root}")
    print(f"files: {stats['count']}")
    print(
        "rms_after_peak_norm: "
        f"mean={stats['mean']:.4f} "
        f"median={stats['median']:.4f} "
        f"p10={stats['p10']:.4f} "
        f"p25={stats['p25']:.4f} "
        f"p75={stats['p75']:.4f} "
        f"p90={stats['p90']:.4f}"
    )
    print(f"suggested_rms_target: {stats['median']:.4f}")


if __name__ == "__main__":
    main()
