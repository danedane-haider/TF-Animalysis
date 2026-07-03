"""
Compute STFT magnitude at an existing f0 contour.
Writes a CSV per file with time, frequency, and magnitude columns.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f0_extraction.extract_f0 import load_precomputed_representation, resolve_precomputed_dir


def _nearest_bin_indices(freqs, target_freqs):
    """Return nearest bin index in freqs for each target frequency."""
    idx = np.searchsorted(freqs, target_freqs)
    idx = np.clip(idx, 1, len(freqs) - 1)
    left = freqs[idx - 1]
    right = freqs[idx]
    choose_right = target_freqs - left > right - target_freqs
    return idx - 1 + choose_right.astype(int)


def compute_f0_magnitudes(input_dir, output_dir_name=None,
                          contour_dir_name="f0_refined",
                          use_precomputed_representations=None,
                          sr=16000, n_fft=8192, hop_length=256):
    """
    Compute STFT magnitudes at f0 contour for all contour files.

    Args:
        input_dir: Directory containing contour CSVs and audio or precomputed STFT
        output_dir_name: Output directory name under input_dir (None = in-place)
        contour_dir_name: Contour directory under input_dir
        use_precomputed_representations: Optional precomputed STFT folder
        sr: Sampling rate used for STFT (audio will be resampled)
        n_fft: FFT size
        hop_length: Hop length in samples
    """
    input_dir = Path(input_dir)
    f0_dir = input_dir / contour_dir_name
    precomputed_dir = (
        resolve_precomputed_dir(use_precomputed_representations, "stft")
        if use_precomputed_representations is not None
        else None
    )

    if not f0_dir.exists():
        print(f"ERROR: contour directory not found: {f0_dir}")
        return

    csv_files = sorted(list(f0_dir.glob("*.f0.csv")))
    if len(csv_files) == 0:
        print(f"ERROR: No .f0.csv files found in {f0_dir}")
        return

    output_dir = None
    if output_dir_name is not None:
        output_dir = input_dir / output_dir_name
        output_dir.mkdir(exist_ok=True)

    print(f"Input:  {f0_dir}")
    print(f"Output: {output_dir if output_dir else f'in-place ({f0_dir.name})'}")
    print(f"Files:  {len(csv_files)}")
    print(f"Source: {precomputed_dir if precomputed_dir else 'computed on the fly'}")
    print(f"Params: sr={sr}, n_fft={n_fft}, hop_length={hop_length}")

    processed = 0
    skipped = 0

    for csv_path in tqdm(csv_files, desc="Computing f0 magnitudes"):
        df = pd.read_csv(csv_path)
        required_cols = {"time", "frequency"}
        if not required_cols.issubset(df.columns):
            print(f"  Skipping {csv_path.name}: missing columns {required_cols - set(df.columns)}")
            skipped += 1
            continue

        stem = csv_path.name[:-len(".f0.csv")] if csv_path.name.endswith(".f0.csv") else csv_path.stem
        audio_path = input_dir / f"{stem}.wav"
        if precomputed_dir is not None:
            try:
                mag, freqs, spec_times = load_precomputed_representation(precomputed_dir, audio_path, "stft")
            except Exception as exc:
                print(f"  Skipping {csv_path.name}: {exc}")
                skipped += 1
                continue
        else:
            if not audio_path.exists():
                print(f"  Skipping {csv_path.name}: audio not found ({audio_path.name})")
                skipped += 1
                continue

            audio, _ = librosa.load(audio_path, sr=sr)
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            mag = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            spec_times = np.arange(mag.shape[1], dtype=float) * hop_length / sr

        times = df["time"].to_numpy()
        f0 = df["frequency"].to_numpy()

        frame_idx = _nearest_bin_indices(spec_times, times)
        valid_frame = (frame_idx >= 0) & (frame_idx < mag.shape[1])
        valid_time = (times >= spec_times[0]) & (times <= spec_times[-1])
        valid_f0 = f0 > 0
        valid = valid_frame & valid_time & valid_f0

        mag_at_f0 = np.zeros_like(f0, dtype=float)
        if np.any(valid):
            freq_idx = _nearest_bin_indices(freqs, f0[valid])
            mag_at_f0[valid] = mag[freq_idx, frame_idx[valid]]

        df["magnitude"] = mag_at_f0

        output_path = csv_path if output_dir is None else output_dir / csv_path.name
        df.to_csv(output_path, index=False)
        processed += 1

    print(f"\n✓ Computed: {processed} files")
    if skipped:
        print(f"⊘ Skipped: {skipped} files")


def main():
    parser = argparse.ArgumentParser(
        description="Compute STFT magnitude at an existing f0 contour."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing contour CSVs and audio or precomputed STFT")
    parser.add_argument("--contour_dir", type=str, default="f0_refined",
                        help="Contour directory under input_dir (default: f0_refined)")
    parser.add_argument("--use_precomputed_representations", type=str, default=None,
                        help="Path to a precomputed STFT folder, or a parent containing stft_*.")
    parser.add_argument("--output_dir_name", type=str, default=None,
                        help="Output directory name under input_dir (default: in-place)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sampling rate for STFT (default: 16000)")
    parser.add_argument("--n_fft", type=int, default=8192,
                        help="FFT size (default: 8192)")
    parser.add_argument("--hop_length", type=int, default=256,
                        help="Hop length in samples (default: 256)")

    args = parser.parse_args()

    compute_f0_magnitudes(
        input_dir=args.input,
        output_dir_name=args.output_dir_name,
        contour_dir_name=args.contour_dir,
        use_precomputed_representations=args.use_precomputed_representations,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )


if __name__ == "__main__":
    main()
