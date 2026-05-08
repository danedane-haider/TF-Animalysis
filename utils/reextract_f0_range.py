"""
Re-extract F0 for a specific range of files with custom parameters.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Import F0 extraction
sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.extract_f0 import extract_f0_elelet


def reextract_f0_range(input_dir, start_idx, end_idx, fmin=22.5, fmax=50, sr=16000, frame_resolution=0.016):
    """
    Re-extract F0 for files in specified range.

    Args:
        input_dir: Directory containing audio files
        start_idx: Start index (1-indexed, inclusive)
        end_idx: End index (1-indexed, inclusive)
        fmin: Minimum frequency for F0 search
        fmax: Maximum frequency for F0 search
        sr: Sample rate
        frame_resolution: Frame resolution in seconds
    """
    input_dir = Path(input_dir)

    # Find all audio files
    audio_files = sorted(list(input_dir.glob("*.wav")))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {input_dir}")

    # Convert to 0-indexed
    start_idx_0 = start_idx - 1
    end_idx_0 = end_idx - 1

    # Validate range
    if start_idx_0 < 0 or end_idx_0 >= len(audio_files):
        raise ValueError(f"Invalid range: {start_idx}-{end_idx}. Total files: {len(audio_files)}")

    # Get files in range
    files_to_process = audio_files[start_idx_0:end_idx_0+1]

    # Create output directory
    f0_dir = input_dir / f"f0_{frame_resolution:.3f}"
    f0_dir.mkdir(exist_ok=True)

    print("="*60)
    print("RE-EXTRACT F0 FOR FILE RANGE")
    print("="*60)
    print(f"Directory: {input_dir}")
    print(f"Output: {f0_dir}")
    print(f"Range: {start_idx}-{end_idx} ({len(files_to_process)} files)")
    print(f"Parameters:")
    print(f"  fmin: {fmin} Hz")
    print(f"  fmax: {fmax} Hz")
    print(f"  sr: {sr} Hz")
    print(f"  frame_resolution: {frame_resolution} s")
    print()

    # Elelet parameters (matching config defaults)
    stride = int(sr * frame_resolution)
    high_pass = 10
    divide_by_2 = False
    max_jump = 1.0
    use_global_peak = True
    energy_threshold = 0.2

    processed = 0
    errors = 0

    for audio_path in tqdm(files_to_process, desc="Extracting F0"):
        try:
            # Load audio
            import soundfile as sf
            audio, _ = sf.read(audio_path)

            # Extract F0
            time, f0, confidence = extract_f0_elelet(
                audio,
                sr=sr,
                stride=stride,
                fmin=fmin,
                fmax=fmax,
                high_pass=high_pass,
                divide_by_2=divide_by_2,
                max_jump=max_jump,
                use_global_peak=use_global_peak,
                energy_threshold=energy_threshold
            )

            # Save to CSV
            output_path = f0_dir / f"{audio_path.stem}.f0.csv"
            df = pd.DataFrame({
                'time': time,
                'frequency': f0,
                'confidence': confidence
            })
            df.to_csv(output_path, index=False)

            processed += 1

        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            errors += 1
            continue

    print()
    print("="*60)
    print("✓ Complete!")
    print("="*60)
    print(f"Processed: {processed} files")
    print(f"Errors: {errors} files")
    print()
    print(f"F0 files saved to: {f0_dir}")
    print()
    print("Next steps:")
    print(f"  1. Review extracted F0 with annotate_f0.py")
    print(f"  2. Copy corrected F0 files if needed")


def main():
    parser = argparse.ArgumentParser(
        description='Re-extract F0 for a specific range of files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Re-extract F0 for files 1778-1847 with custom frequency range
  python preprocessing/reextract_f0_range.py --input data/ntvow_low --start 1778 --end 1847 --fmin 20 --fmax 40

Note: File numbers are based on alphabetically sorted file list (1-indexed).
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing audio files')
    parser.add_argument('--start', type=int, required=True,
                        help='Start file index (1-indexed, inclusive)')
    parser.add_argument('--end', type=int, required=True,
                        help='End file index (1-indexed, inclusive)')
    parser.add_argument('--fmin', type=float, default=22.5,
                        help='Minimum frequency for F0 search (default: 22.5 Hz)')
    parser.add_argument('--fmax', type=float, default=50,
                        help='Maximum frequency for F0 search (default: 50 Hz)')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate (default: 16000)')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution in seconds (default: 0.016)')

    args = parser.parse_args()

    reextract_f0_range(
        input_dir=args.input,
        start_idx=args.start,
        end_idx=args.end,
        fmin=args.fmin,
        fmax=args.fmax,
        sr=args.sr,
        frame_resolution=args.frame_resolution
    )


if __name__ == "__main__":
    main()
