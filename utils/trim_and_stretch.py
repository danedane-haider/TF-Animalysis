"""
Trim audio files to markers in f0_corrected and stretch to 2.5s if needed.

This script:
1. Reads start_point and end_point markers from f0_corrected CSV files
2. Trims audio to the marked region
3. Stretches audio to 2.5s if it's shorter (maintains pitch)
4. Saves trimmed/stretched audio to output directory
"""

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm


def trim_and_stretch_audio(audio_path, f0_csv_path, sr=16000, target_duration=2.5):
    """
    Trim audio to markers and stretch if needed.

    Args:
        audio_path: Path to audio file
        f0_csv_path: Path to f0_corrected CSV file with markers
        sr: Sample rate
        target_duration: Target duration in seconds

    Returns:
        processed_audio: Trimmed and possibly stretched audio
        trimmed_f0_df: Trimmed and adjusted F0 dataframe
        trim_start: Start time of trim (seconds)
        trim_end: End time of trim (seconds)
        was_stretched: Whether stretching was applied
        stretch_rate: Time stretch factor (1.0 = no stretch)
    """
    # Load audio
    audio, file_sr = sf.read(audio_path)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    # Load F0 CSV with markers
    df = pd.read_csv(f0_csv_path)

    # Check if markers exist
    if 'start_point' in df.columns and 'end_point' in df.columns:
        # Find start and end markers
        start_idx = df[df['start_point'] == 1].index
        end_idx = df[df['end_point'] == 1].index

        if len(start_idx) == 0 or len(end_idx) == 0:
            # No markers set, use entire audio
            start_time = 0.0
            end_time = len(audio) / sr
            start_frame_idx = 0
            end_frame_idx = len(df)
        else:
            # Get start and end times
            start_time = df.loc[start_idx[0], 'time']
            end_time = df.loc[end_idx[0], 'time']
            start_frame_idx = start_idx[0]
            end_frame_idx = end_idx[0] + 1  # Include end frame
    else:
        # No marker columns, use entire audio
        start_time = 0.0
        end_time = len(audio) / sr
        start_frame_idx = 0
        end_frame_idx = len(df)

    # Convert to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Trim audio
    trimmed_audio = audio[start_sample:end_sample]
    trimmed_duration = len(trimmed_audio) / sr

    # Trim F0 dataframe
    trimmed_f0_df = df.iloc[start_frame_idx:end_frame_idx].copy()

    # Reset time to start from 0
    trimmed_f0_df['time'] = trimmed_f0_df['time'] - start_time

    # Stretch if needed
    was_stretched = False
    stretch_rate = 1.0

    if target_duration is not None and trimmed_duration < target_duration:
        # Calculate stretch rate to reach target duration
        stretch_rate = trimmed_duration / target_duration
        stretched_audio = librosa.effects.time_stretch(trimmed_audio, rate=stretch_rate)
        was_stretched = True
        processed_audio = stretched_audio

        # Stretch F0 timestamps proportionally
        trimmed_f0_df['time'] = trimmed_f0_df['time'] / stretch_rate
    else:
        processed_audio = trimmed_audio

    return processed_audio, trimmed_f0_df, start_time, end_time, was_stretched, stretch_rate


def process_dataset(input_dir, output_dir, f0_dir=None, target_duration=2.5, skip_existing=True):
    """Process entire dataset with trimming and stretching."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create F0 output directory (named f0_corrected for training compatibility)
    f0_output_dir = output_dir / "f0_corrected"
    f0_output_dir.mkdir(exist_ok=True, parents=True)

    # Find F0 directory
    if f0_dir is None:
        f0_dir = input_dir / "f0_corrected"
    else:
        f0_dir = Path(f0_dir)

    if not f0_dir.exists():
        raise ValueError(f"F0 directory not found: {f0_dir}")

    # Find all audio files
    audio_files = sorted(list(input_dir.glob("*.wav")))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {input_dir}")

    print("="*60)
    print("TRIM AND STRETCH NTVOW VOWELS")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"F0 input directory: {f0_dir}")
    print(f"F0 output directory: {f0_output_dir}")
    print(f"Files to process: {len(audio_files)}")
    print(f"Target duration: {target_duration}s")
    print(f"Skip existing: {skip_existing}")
    print()

    processed = 0
    skipped = 0
    errors = 0
    stretched_count = 0
    trim_durations = []
    stretch_rates = []

    for audio_path in tqdm(audio_files, desc="Processing"):
        output_path = output_dir / audio_path.name

        if skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            # Find corresponding F0 file
            f0_path = f0_dir / f"{audio_path.stem}.f0.csv"
            if not f0_path.exists():
                print(f"\n  Warning: F0 file not found for {audio_path.name}, skipping")
                skipped += 1
                continue

            # Process audio and F0
            processed_audio, trimmed_f0_df, start_time, end_time, was_stretched, stretch_rate = trim_and_stretch_audio(
                audio_path, f0_path, target_duration=target_duration
            )

            # Save processed audio
            sf.write(output_path, processed_audio, 16000)

            # Save trimmed F0
            f0_output_path = f0_output_dir / f"{audio_path.stem}.f0.csv"
            trimmed_f0_df.to_csv(f0_output_path, index=False)

            # Statistics
            trim_duration = end_time - start_time
            trim_durations.append(trim_duration)
            if was_stretched:
                stretched_count += 1
                stretch_rates.append(stretch_rate)

            processed += 1

        except Exception as e:
            print(f"\n  Error processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
            continue

    print()
    print("="*60)
    print("✓ Trim and stretch complete!")
    print("="*60)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Stretched: {stretched_count} ({100*stretched_count/max(processed,1):.1f}%)")
    print()
    if len(trim_durations) > 0:
        print("Trim statistics:")
        print(f"  Mean trim duration: {np.mean(trim_durations):.3f}s")
        print(f"  Min trim duration: {np.min(trim_durations):.3f}s")
        print(f"  Max trim duration: {np.max(trim_durations):.3f}s")
        print()
    if len(stretch_rates) > 0:
        print("Stretch statistics:")
        print(f"  Mean stretch rate: {np.mean(stretch_rates):.3f}")
        print(f"  Min stretch rate: {np.min(stretch_rates):.3f}")
        print(f"  Max stretch rate: {np.max(stretch_rates):.3f}")
        print()
    print(f"Processed audio saved to: {output_dir}")
    print(f"Trimmed F0 files saved to: {f0_output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Verify a few samples to ensure trimming is correct")
    print(f"  2. Apply formant filtering to this trimmed dataset")
    print(f"  3. F0 files in f0_corrected/ are ready for training (timestamps start from 0)")


def main():
    parser = argparse.ArgumentParser(
        description='Trim audio to markers and stretch to target duration if needed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python preprocessing/trim_and_stretch.py --input data/ntvow_low --output data/ntvow_low_trimmed

What it does:
  1. Reads start_point and end_point markers from f0_corrected/*.f0.csv
  2. Trims audio to the marked region (start_point=1 to end_point=1)
  3. Trims F0 files and resets timestamps to start from 0
  4. If trimmed audio < 2.5s, stretches it to 2.5s (maintains pitch)
  5. Saves processed audio and F0 files to output/f0_corrected/

This ensures all vowels are at least 2.5s long and F0 files match audio exactly.
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (should contain audio + f0_corrected/)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed audio')
    parser.add_argument('--f0_dir', type=str, default=None,
                        help='F0 directory path (default: input/f0_corrected)')
    parser.add_argument('--target_duration', type=float, default=None,
                        help='Target duration in seconds (default: 2.5)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip files that already exist (default: True)')
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false',
                        help='Reprocess all files')

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        f0_dir=args.f0_dir,
        target_duration=args.target_duration,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
