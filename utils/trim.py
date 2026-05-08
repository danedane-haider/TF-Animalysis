"""
Trim audio files based on start/end markers in corrected f0 files
Cuts audio to keep only the region between start and end points
"""

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def trim_audio_by_markers(audio_dir, frame_resolution=0.016, output_suffix="_trimmed"):
    """
    Trim audio files based on start/end markers in corrected f0 files

    Args:
        audio_dir: Directory with audio files, f0_corrected/ subdirectory
        frame_resolution: Frame resolution to find f0 directory
        output_suffix: Suffix to add to output directory name
    """
    audio_dir = Path(audio_dir)
    corrected_dir = audio_dir / "f0_corrected"

    if not corrected_dir.exists():
        print(f"ERROR: Corrected f0 directory not found: {corrected_dir}")
        print("Please run correct_f0_interactive.py first")
        return

    # Create output directory
    output_dir = Path(str(audio_dir) + output_suffix)
    output_dir.mkdir(exist_ok=True)

    # Find all corrected f0 files
    f0_files = sorted(list(corrected_dir.glob("*.f0.csv")))

    if len(f0_files) == 0:
        print(f"ERROR: No corrected f0 files found in {corrected_dir}")
        return

    print("="*60)
    print("TRIM AUDIO BY START/END MARKERS")
    print("="*60)
    print(f"Input:  {audio_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(f0_files)} corrected f0 files")
    print("="*60 + "\n")

    successful = 0
    skipped = 0
    failed = []

    for f0_path in tqdm(f0_files, desc="Trimming audio files"):
        try:
            # Load f0 data
            df = pd.read_csv(f0_path)

            # Check if start/end points exist
            if 'start_point' not in df.columns or 'end_point' not in df.columns:
                print(f"\n  Skipping {f0_path.stem}: No start/end markers")
                skipped += 1
                continue

            # Find start and end times
            start_mask = df['start_point'] == 1
            end_mask = df['end_point'] == 1

            if not start_mask.any() or not end_mask.any():
                print(f"\n  Skipping {f0_path.stem}: Missing start or end marker")
                skipped += 1
                continue

            t_start = df.loc[start_mask, 'time'].values[0]
            t_end = df.loc[end_mask, 'time'].values[0]

            # Ensure start < end
            if t_start > t_end:
                t_start, t_end = t_end, t_start

            # Find corresponding audio file
            audio_path = audio_dir / f"{f0_path.stem.replace('.f0', '')}.wav"
            if not audio_path.exists():
                print(f"\n  Warning: Audio file not found: {audio_path.name}")
                failed.append(f0_path.name)
                continue

            # Load audio
            audio, sr = sf.read(str(audio_path))

            # Convert times to sample indices
            start_sample = int(t_start * sr)
            end_sample = int(t_end * sr)

            # Clip to valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            # Trim audio
            trimmed_audio = audio[start_sample:end_sample]

            # Save trimmed audio
            output_path = output_dir / audio_path.name
            sf.write(str(output_path), trimmed_audio, sr)

            # Also save the trimmed f0 file
            output_f0_dir = output_dir / f"f0_{frame_resolution:.3f}"
            output_f0_dir.mkdir(exist_ok=True)

            # Trim f0 data to match
            time_mask = (df['time'] >= t_start) & (df['time'] <= t_end)
            df_trimmed = df[time_mask].copy()

            # Adjust time to start from 0
            df_trimmed['time'] = df_trimmed['time'] - t_start

            # Save trimmed f0
            output_f0_path = output_f0_dir / f0_path.name
            df_trimmed.to_csv(output_f0_path, index=False)

            successful += 1

        except Exception as e:
            print(f"\n  Error processing {f0_path.name}: {e}")
            failed.append(f0_path.name)
            continue

    print("\n" + "="*60)
    print("TRIMMING COMPLETE")
    print("="*60)
    print(f"✓ Successfully trimmed: {successful}/{len(f0_files)} files")
    if skipped > 0:
        print(f"⊘ Skipped (no markers): {skipped} files")
    if failed:
        print(f"✗ Failed: {len(failed)} files")
        for f in failed:
            print(f"  - {f}")

    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── *.wav (trimmed audio)")
    print(f"    └── f0_{frame_resolution:.3f}/*.csv (trimmed f0)")

    if successful > 0:
        print(f"\n✓ Trimmed audio saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Check trimmed audio files")
        print(f"  2. Use this directory for training:")
        print(f"     python split_dataset.py --input {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Trim audio files based on start/end markers in corrected f0 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trim all files with corrected f0 markers
  python trim_audio_by_markers.py --input data/rumbles

  # Specify output directory name
  python trim_audio_by_markers.py --input data/rumbles --output_suffix "_isolated"

Workflow:
  1. Run correct_f0_interactive.py and mark start/end for each file
  2. Run this script to trim audio files
  3. Use trimmed files for training (they will all be isolated rumbles)
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with audio files and f0_corrected/ subdirectory')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution (default: 0.016)')
    parser.add_argument('--output_suffix', type=str, default='_trimmed',
                        help='Suffix for output directory (default: _trimmed)')

    args = parser.parse_args()

    trim_audio_by_markers(
        audio_dir=args.input,
        frame_resolution=args.frame_resolution,
        output_suffix=args.output_suffix
    )


if __name__ == "__main__":
    main()
