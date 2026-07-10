"""
Pre-compute STFT and Elelet representations for all audio files
Stores results in data/rumbles/stft/ and data/rumbles/elelet/
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

import librosa
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from f0_extraction.extract_f0 import METAINFO_FILENAME
from f0_extraction.pipeline import representation_dir


def write_metainfo(output_dir, representation, parameters):
    """Write the transform parameters used for this representation folder."""
    meta_path = Path(output_dir) / METAINFO_FILENAME
    meta = {
        "representation": representation,
        "parameters": parameters,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def amplitude_to_db(magnitude):
    magnitude = np.asarray(magnitude)
    amin = 1e-5
    ref = max(float(np.max(magnitude)), amin)
    db = 20.0 * np.log10(np.maximum(amin, magnitude)) - 20.0 * np.log10(ref)
    return np.maximum(db, np.max(db) - 80.0)


def precompute_stft(audio_dir, output_dir, sr=16000, n_fft=8192, hop_length=256,
                    win_length=None, skip_existing=True, start_idx=0):
    """Pre-compute and save STFT representations for all audio files"""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    parameters = {
        "sr": sr,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "window": "hann",
        "center": True,
    }
    write_metainfo(output_dir, "stft", parameters)

    # Find all audio files
    audio_files = sorted(list(audio_dir.glob("*.wav")))

    print(f"\nPre-computing STFT representations...")
    print(f"  Audio directory: {audio_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files to process: {len(audio_files)}")
    print(f"  Starting from index: {start_idx}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Parameters: n_fft={n_fft}, win_length={win_length}, hop_length={hop_length}, sr={sr}")

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
            D = librosa.stft(
                audio,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window="hann",
                center=True,
            )
            magnitude = np.abs(D)
            S_db = amplitude_to_db(magnitude)
            spec_times = librosa.times_like(magnitude, sr=sr, hop_length=hop_length)
            spec_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Save to .npz file
            np.savez_compressed(
                output_path,
                magnitude=magnitude,
                S_db=S_db,
                times=spec_times,
                freqs=spec_freqs,
                sr=sr,
                n_fft=n_fft,
                win_length=-1 if win_length is None else win_length,
                hop_length=hop_length
            )
            processed += 1
        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            print(f"Continuing from next file...")

    print(f"✓ STFT representations saved to {output_dir}")
    print(f"  Processed: {processed}, Skipped: {skipped}")


def precompute_elelet(audio_dir, output_dir, sr=16000, hop_length=256,
                      num_channels=2048, kernel_size=24000, f_max=100, f_min=10,
                      supp_mult=0.2, scale="elelog", backend="fft_decimated",
                      channel_batch_size=None, skip_existing=True, start_idx=0):
    """Pre-compute and save Elelet representations for all audio files"""
    import torch
    from tf_representations.transforms import Elelet

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    parameters = {
        "sr": sr,
        "stride": hop_length,
        "kernel_size": kernel_size,
        "num_channels": num_channels,
        "f_min": f_min,
        "f_max": f_max,
        "supp_mult": supp_mult,
        "scale": scale,
        "use_torch": False,
        "backend": backend,
        "channel_batch_size": channel_batch_size,
        "cache_kernel_fft": False,
    }
    write_metainfo(output_dir, "elelet", parameters)

    # Initialize Elelet transform
    elelet_transform = Elelet(
        kernel_size=kernel_size,
        num_channels=num_channels,
        stride=hop_length,
        f_max=f_max,
        fs=sr,
        supp_mult=supp_mult,
        f_min=f_min,
        scale=scale,
        use_torch=False,  # Use numpy for faster batch processing
        backend=backend,
        channel_batch_size=channel_batch_size,
        cache_kernel_fft=False,  # File lengths vary, so the spectra are rarely reusable.
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
    print(f"  Parameters: num_channels={num_channels}, kernel_size={kernel_size}, stride={hop_length}, f_max={f_max}Hz, f_min={f_min}Hz")

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
                kernel_size=kernel_size,
                num_channels=num_channels,
                f_max=f_max,
                f_min=f_min,
                supp_mult=supp_mult,
                scale=scale
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
    parser.add_argument('--win_length', type=int, default=None,
                        help='Window length for STFT (default: n_fft)')
    parser.add_argument('--hop_length', type=int, default=320,
                        help='Hop length for both STFT and Elelet (default: 320)')
    parser.add_argument('--f_max', type=float, default=500,
                        help='Maximum frequency label for precomputed folders and Elelet f_max (Hz, default: 750)')
    parser.add_argument('--num_channels', type=int, default=1024,
                        help='Number of channels for Elelet (default: 1024)')
    parser.add_argument('--kernel_size', type=int, default=16000,
                        help='Kernel size for Elelet (default: 24000)')
    parser.add_argument('--f_min', type=float, default=5,
                        help='Minimum frequency for Elelet (Hz, default: 10)')
    parser.add_argument('--supp_mult', type=float, default=0.3,
                        help='Support multiplier for Elelet (default: 0.3)')
    parser.add_argument('--scale', type=str, default='elelog',
                        help='Frequency scale for Elelet (default: elelog)')
    parser.add_argument('--elelet_backend', choices=('fft_decimated', 'fft'),
                        default='fft_decimated',
                        help='Elelet convolution backend (default: fft_decimated)')
    parser.add_argument('--elelet_channel_batch_size', type=int, default=None,
                        help='Optional Elelet channel batch size to limit memory')
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
    stft_dir = audio_dir / representation_dir("stft", args.f_max)
    elelet_dir = audio_dir / representation_dir("elelet", args.f_max)

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
            win_length=args.win_length,
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
            kernel_size=args.kernel_size,
            f_max=args.f_max,
            f_min=args.f_min,
            supp_mult=args.supp_mult,
            scale=args.scale,
            backend=args.elelet_backend,
            channel_batch_size=args.elelet_channel_batch_size,
            skip_existing=args.skip_existing,
            start_idx=args.start_idx
        )

    print("\n" + "="*60)
    print("✓ Pre-computation complete!")
    print("="*60)
    print("\nYou can now use extraction scripts with --use_precomputed_representations")
    print("Examples:")
    if not args.elelet_only:
        print(f"  python f0_extraction/extract_f0_stft.py --input {audio_dir} --use_precomputed_representations {stft_dir}")
    if not args.stft_only:
        print(f"  python f0_extraction/extract_f0_elelet.py --input {audio_dir} --use_precomputed_representations {elelet_dir}")


if __name__ == "__main__":
    main()
