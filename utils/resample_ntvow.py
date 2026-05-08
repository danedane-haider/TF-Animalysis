"""
Resample NTVOW vowel dataset using the resampling trick to shift pitch down by 8x.

The trick:
1. Load audio at original sample rate (e.g., 48kHz with F0 ~100-200 Hz)
2. Resample to 128kHz (16kHz * 8) - creates more samples at intermediate rate
3. Save with 16kHz metadata - when played at 16kHz, it's 8x slower
4. Result: F0 drops by 8x (100 Hz → 12.5 Hz, perfect for rumbles!)

By default, this also time-stretches by 8x (1 sec → 8 sec).

Optional: Use --time_stretch_factor to decouple pitch and time using phase vocoder.
Example: --time_stretch_factor 2 gives 8x pitch shift but only 4x time stretch (1 sec → 4 sec).
"""

import argparse
import numpy as np
import soundfile as sf
import torchaudio
import torch
import librosa
from pathlib import Path
from tqdm import tqdm


def resample_vowels(input_dir, output_dir, target_sr=16000, upsample_factor=8,
                    time_stretch_factor=None, skip_existing=True):
    """
    Resample vowel dataset using pitch-shift trick with optional time stretch decoupling.

    Args:
        input_dir: Path to NTVOW dataset
        output_dir: Path to save resampled files (will create if doesn't exist)
        target_sr: Target sample rate that will be used for playback (default: 16000)
        upsample_factor: Resampling factor for pitch shift (default: 8 = 3 octaves down)
        time_stretch_factor: If provided, apply phase vocoder to decouple pitch and time.
                           Final time stretch = upsample_factor / time_stretch_factor.
                           Example: upsample_factor=8, time_stretch_factor=2 → 8x pitch, 4x time
                           If None, pitch and time are coupled (both = upsample_factor)
        skip_existing: Skip files that already exist in output
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.WAV', '*.flac', '*.FLAC', '*.mp3', '*.MP3']:
        audio_files.extend(list(input_dir.glob(ext)))
    audio_files = sorted(audio_files)

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {input_dir}")

    print("="*60)
    print("RESAMPLE NTVOW VOWELS FOR ELEPHANT RUMBLE SYNTHESIS")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(audio_files)}")
    print(f"Upsample factor: {upsample_factor}x (pitch down by {upsample_factor}x)")
    if time_stretch_factor is not None:
        actual_time_stretch = upsample_factor / time_stretch_factor
        print(f"Time stretch factor: {time_stretch_factor}x (decoupled)")
        print(f"Final time stretch: {actual_time_stretch:.1f}x")
        print(f"Method: Resampling trick + phase vocoder")
    else:
        print(f"Time stretch: {upsample_factor}x (coupled with pitch)")
        print(f"Method: Resampling trick only")
    print(f"Target sample rate: {target_sr} Hz")
    print(f"Skip existing: {skip_existing}")
    print()

    processed = 0
    skipped = 0
    errors = 0

    for audio_path in tqdm(audio_files, desc="Resampling vowels"):
        # Check if already exists
        output_path = output_dir / f"{audio_path.stem}.wav"
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            # Load audio
            x, orig_fs = sf.read(audio_path)

            # Convert to torch tensor [channels, samples]
            if x.ndim == 1:
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            else:
                x = torch.tensor(x, dtype=torch.float32).T  # [samples, channels] -> [channels, samples]

            # Only use first channel if stereo
            if x.shape[0] > 1:
                x = x[0:1, :]

            # Apply resampling trick: resample to intermediate rate
            # Intermediate rate = target_sr * upsample_factor (e.g., 16kHz * 8 = 128kHz)
            # This ensures consistent pitch shift regardless of input sample rate
            intermediate_sr = target_sr * upsample_factor
            resampler = torchaudio.transforms.Resample(
                orig_freq=int(orig_fs),
                new_freq=int(intermediate_sr)
            )
            x_resampled = resampler(x)

            # Convert back to numpy
            x_resampled = x_resampled.squeeze(0).numpy()

            # Apply phase vocoder time stretch if requested (decouples pitch and time)
            if time_stretch_factor is not None:
                # Phase vocoder: change duration without changing pitch
                # rate > 1.0 speeds up (shorter), rate < 1.0 slows down (longer)
                # We want to speed up by time_stretch_factor to partially undo the time stretch
                rate = time_stretch_factor  # e.g., 2.0 = speed up by 2x (halve duration)
                x_resampled = librosa.effects.time_stretch(x_resampled, rate=rate)

            # Save with target_sr metadata (this is the trick!)
            # The audio has 8x more samples, but we label it as target_sr
            # When played at target_sr, it plays 8x slower (pitch down by 8x)
            sf.write(output_path, x_resampled, target_sr)

            processed += 1

        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            errors += 1
            continue

    print()
    print("="*60)
    print("✓ Resampling complete!")
    print("="*60)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print()
    print(f"Resampled vowels saved to: {output_dir}")
    print()
    print("Example transformations (assuming original F0 ~100-200 Hz, 1 sec duration):")
    if time_stretch_factor is not None:
        actual_time = 1.0 * upsample_factor / time_stretch_factor
        print(f"  Pitch: 100 Hz → {100/upsample_factor:.1f} Hz, 150 Hz → {150/upsample_factor:.1f} Hz, 200 Hz → {200/upsample_factor:.1f} Hz")
        print(f"  Duration: 1.0 sec → {actual_time:.1f} sec")
    else:
        print(f"  Pitch: 100 Hz → {100/upsample_factor:.1f} Hz, 150 Hz → {150/upsample_factor:.1f} Hz, 200 Hz → {200/upsample_factor:.1f} Hz")
        print(f"  Duration: 1.0 sec → {1.0*upsample_factor:.1f} sec")
    print()
    print("Next steps:")
    print(f"  1. Extract F0: python preprocessing/extract_f0.py --input {output_dir}")
    print(f"  2. Annotate (optional): python preprocessing/annotate_f0.py --input {output_dir}")
    print(f"  3. Update config to include {output_dir} in training")


def main():
    parser = argparse.ArgumentParser(
        description='Resample NTVOW vowels using pitch-shift trick for elephant rumble synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: 8x pitch shift, 8x time stretch (1 sec → 8 sec)
  python preprocessing/resample_ntvow.py --input /path/to/NTVOW --output data/ntvow

  # Advanced: 8x pitch shift, 4x time stretch (1 sec → 4 sec)
  python preprocessing/resample_ntvow.py --input /path/to/NTVOW --output data/ntvow --time_stretch_factor 2

The resampling trick:
  - Resamples to intermediate rate (e.g., 48kHz → 128kHz)
  - Saves with 16kHz metadata (lies about sample rate)
  - When loaded/played at 16kHz, plays 8x slower
  - Result: Pitch drops by 8x (100 Hz → 12.5 Hz)
  - Time stretches by 8x (1 sec → 8 sec)

Decoupling pitch and time (optional):
  - Use --time_stretch_factor to apply phase vocoder after resampling
  - Example: --time_stretch_factor 2 speeds up by 2x (preserves pitch)
  - Result: 8x pitch shift, 4x time stretch (1 sec → 4 sec)
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Path to NTVOW dataset directory')
    parser.add_argument('--output', type=str, default='data/ntvow',
                        help='Output directory for resampled vowels (default: data/ntvow)')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='Target sample rate for output files (default: 16000)')
    parser.add_argument('--upsample_factor', type=int, default=8,
                        help='Upsample factor for pitch shift (default: 8 = 3 octaves down)')
    parser.add_argument('--time_stretch_factor', type=float, default=None,
                        help='Time stretch factor to decouple pitch and time. '
                             'Final time stretch = upsample_factor / time_stretch_factor. '
                             'Example: --upsample_factor 8 --time_stretch_factor 2 gives 8x pitch, 4x time. '
                             'If not provided, pitch and time are coupled (both = upsample_factor)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip files that already exist (default: True)')
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false',
                        help='Recompute all files, overwriting existing')

    args = parser.parse_args()

    resample_vowels(
        input_dir=args.input,
        output_dir=args.output,
        target_sr=args.target_sr,
        upsample_factor=args.upsample_factor,
        time_stretch_factor=args.time_stretch_factor,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
