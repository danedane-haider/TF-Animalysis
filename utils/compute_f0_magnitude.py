"""
Compute STFT magnitude at the f0 contour specified in f0_corrected CSVs.
Writes a CSV per file with time, frequency, and magnitude columns.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def _nearest_bin_indices(freqs, target_freqs):
    """Return nearest bin index in freqs for each target frequency."""
    idx = np.searchsorted(freqs, target_freqs)
    idx = np.clip(idx, 1, len(freqs) - 1)
    left = freqs[idx - 1]
    right = freqs[idx]
    choose_right = target_freqs - left > right - target_freqs
    return idx - 1 + choose_right.astype(int)


def compute_f0_magnitudes(input_dir, output_dir_name=None,
                          sr=16000, n_fft=8192, hop_length=256):
    """
    Compute STFT magnitudes at f0 contour for all corrected f0 files.

    Args:
        input_dir: Directory containing f0_corrected/ and audio files
        output_dir_name: Output directory name under input_dir (None = in-place)
        sr: Sampling rate used for STFT (audio will be resampled)
        n_fft: FFT size
        hop_length: Hop length in samples
    """
    input_dir = Path(input_dir)
    f0_dir = input_dir / "f0_corrected"

    if not f0_dir.exists():
        print(f"ERROR: f0_corrected directory not found: {f0_dir}")
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
    print(f"Output: {output_dir if output_dir else 'in-place (f0_corrected)'}")
    print(f"Files:  {len(csv_files)}")
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

        audio_path = input_dir / f"{csv_path.stem.replace('.f0', '')}.wav"
        if not audio_path.exists():
            print(f"  Skipping {csv_path.name}: audio not found ({audio_path.name})")
            skipped += 1
            continue

        audio, _ = librosa.load(audio_path, sr=sr)
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        times = df["time"].to_numpy()
        f0 = df["frequency"].to_numpy()

        frame_idx = np.rint(times * sr / hop_length).astype(int)
        valid_frame = (frame_idx >= 0) & (frame_idx < mag.shape[1])
        valid_f0 = f0 > 0
        valid = valid_frame & valid_f0

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
        description="Compute STFT magnitude at the f0 contour from f0_corrected CSVs"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing f0_corrected/ and audio files")
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
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )


if __name__ == "__main__":
    main()
