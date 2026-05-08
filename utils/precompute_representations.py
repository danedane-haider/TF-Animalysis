"""
Pre-compute STFT and Elelet representations for all audio files
Stores results in data/rumbles/stft/ and data/rumbles/elelet/
"""

import argparse
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Import Elelet transform
sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.transforms import Elelet
import torch


def precompute_stft(audio_dir, output_dir, sr=16000, n_fft=8192, hop_length=256,
                    skip_existing=True, start_idx=0):
    """Pre-compute and save STFT representations for all audio files"""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all audio files
    audio_files = sorted(list(audio_dir.glob("*.wav")))

    print(f"\nPre-computing STFT representations...")
    print(f"  Audio directory: {audio_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files to process: {len(audio_files)}")
    print(f"  Starting from index: {start_idx}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Parameters: n_fft={n_fft}, hop_length={hop_length}, sr={sr}")

    processed = 0
    skipped = 0

    for i, audio_path in enumerate(tqdm(audio_files[start_idx:], desc="STFT", initial=start_idx, total=len(audio_files))):
        # Check if already exists
        output_path = output_dir / f"{audio_path.stem}.npz"
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=sr)

            # Compute STFT
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            spec_times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
            spec_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Save to .npz file
            np.savez_compressed(
                output_path,
                S_db=S_db,
                times=spec_times,
                freqs=spec_freqs,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            processed += 1
        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            print(f"Continuing from next file...")

    print(f"✓ STFT representations saved to {output_dir}")
    print(f"  Processed: {processed}, Skipped: {skipped}")


def precompute_elelet(audio_dir, output_dir, sr=16000, hop_length=256,
                      num_channels=2048, fmax=100, fmin=10, skip_existing=True, start_idx=0):
    """Pre-compute and save Elelet representations for all audio files"""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize Elelet transform
    elelet_transform = Elelet(
        kernel_size=24000,
        num_channels=num_channels,
        stride=hop_length,
        fmax=fmax,
        fs=sr,
        supp_mult=0.2,
        fmin=fmin,
        scale='elelog',
        use_torch=False,  # Use numpy for faster batch processing
    )

    # Convert fc to numpy
    if isinstance(elelet_transform.fc, torch.Tensor):
        elelet_transform.fc = elelet_transform.fc.numpy()

    # Find all audio files
    audio_files = sorted(list(audio_dir.glob("*.wav")))

    print(f"\nPre-computing Elelet representations...")
    print(f"  Audio directory: {audio_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files to process: {len(audio_files)}")
    print(f"  Starting from index: {start_idx}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Parameters: num_channels={num_channels}, stride={hop_length}, fmax={fmax}Hz, fmin={fmin}Hz")

    processed = 0
    skipped = 0

    for i, audio_path in enumerate(tqdm(audio_files[start_idx:], desc="Elelet", initial=start_idx, total=len(audio_files))):
        # Check if already exists
        output_path = output_dir / f"{audio_path.stem}.npz"
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=sr)

            # Compute Elelet
            elelet_coeffs = elelet_transform(audio)
            elelet_coeffs_abs = np.abs(elelet_coeffs)
            num_frames = elelet_coeffs.shape[1]
            elelet_times = np.arange(num_frames) * elelet_transform.stride / sr

            # Save to .npz file (save both real and abs for flexibility)
            np.savez_compressed(
                output_path,
                coeffs=elelet_coeffs,
                coeffs_abs=elelet_coeffs_abs,
                times=elelet_times,
                fc=elelet_transform.fc,
                sr=sr,
                stride=hop_length,
                num_channels=num_channels,
                fmax=fmax,
                fmin=fmin
            )
            processed += 1
        except Exception as e:
            print(f"\nError processing {audio_path.name} (index {start_idx + i}): {e}")
            print(f"To resume from this file, use: --start_idx {start_idx + i}")
            print(f"Continuing from next file...")

    print(f"✓ Elelet representations saved to {output_dir}")
    print(f"  Processed: {processed}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute STFT and Elelet representations for all audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with audio files (e.g., data/rumbles)')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate (default: 16000)')
    parser.add_argument('--n_fft', type=int, default=8192,
                        help='FFT size for STFT (default: 8192)')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop length for both STFT and Elelet (default: 256)')
    parser.add_argument('--fmax', type=float, default=750,
                        help='Maximum frequency for Elelet (Hz, default: 100)')
    parser.add_argument('--num_channels', type=int, default=1024,
                        help='Number of channels for Elelet (default: 2048)')
    parser.add_argument('--fmin', type=float, default=10,
                        help='High-pass cutoff for Elelet (Hz, default: 10)')
    parser.add_argument('--stft_only', action='store_true',
                        help='Only compute STFT representations')
    parser.add_argument('--elelet_only', action='store_true',
                        help='Only compute Elelet representations')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip files that already have computed representations (default: True)')
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false',
                        help='Recompute all files, overwriting existing')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start processing from this file index (0-based, default: 0)')

    args = parser.parse_args()

    audio_dir = Path(args.input)
    stft_dir = audio_dir / "stft_750"
    elelet_dir = audio_dir / "elelet_750"

    print("="*60)
    print("PRE-COMPUTE REPRESENTATIONS")
    print("="*60)

    # Compute STFT unless elelet_only is specified
    if not args.elelet_only:
        precompute_stft(
            audio_dir=audio_dir,
            output_dir=stft_dir,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            skip_existing=args.skip_existing,
            start_idx=args.start_idx
        )

    # Compute Elelet unless stft_only is specified
    if not args.stft_only:
        precompute_elelet(
            audio_dir=audio_dir,
            output_dir=elelet_dir,
            sr=args.sr,
            hop_length=args.hop_length,
            num_channels=args.num_channels,
            fmax=args.fmax,
            fmin=args.fmin,
            skip_existing=args.skip_existing,
            start_idx=args.start_idx
        )

    print("\n" + "="*60)
    print("✓ Pre-computation complete!")
    print("="*60)
    print("\nYou can now use annotate_f0.py with --precomputed flag")
    print("Example:")
    print(f"  python annotate_f0.py --input {audio_dir} --precomputed")


if __name__ == "__main__":
    main()
