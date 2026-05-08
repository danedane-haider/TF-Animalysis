"""
Prepare training data by organizing corrected samples into train/test splits
"""

import os
import shutil
from pathlib import Path
import random
import argparse


def prepare_training_data(
    audio_dir="data/rumbles",
    corrected_f0_dir="data/rumbles/f0_corrected",
    output_dir="data/rumbles",
    frame_resolution=0.016
):
    """
    Organize corrected samples into train directory

    Args:
        audio_dir: Directory with WAV files
        corrected_f0_dir: Directory with corrected f0 CSV files
        output_dir: Base output directory (will create train/ subdir)
        frame_resolution: Frame resolution for f0 directory name
    """
    audio_dir = Path(audio_dir)
    corrected_f0_dir = Path(corrected_f0_dir)
    output_dir = Path(output_dir)

    # Create output directories
    train_dir = output_dir
    train_f0_dir = train_dir / f"f0_{frame_resolution:.3f}"

    train_dir.mkdir(exist_ok=True)
    train_f0_dir.mkdir(exist_ok=True)

    # Find all corrected f0 files
    corrected_files = sorted(list(corrected_f0_dir.glob("*.f0.csv")))

    if len(corrected_files) == 0:
        print(f"ERROR: No corrected f0 files found in {corrected_f0_dir}")
        return

    print(f"Found {len(corrected_files)} corrected samples")

    # Extract audio filenames from f0 filenames
    # f0 filename format: "FILENAME.f0.csv" -> audio is "FILENAME.wav"
    audio_files = []
    for f0_file in corrected_files:
        # Remove .f0.csv suffix to get base filename
        base_name = f0_file.stem.replace('.f0', '')
        audio_file = audio_dir / f"{base_name}.wav"
        if audio_file.exists():
            audio_files.append(audio_file)
        else:
            print(f"WARNING: Audio file not found: {audio_file}")

    print(f"Found {len(audio_files)} matching audio files")
    print("="*60)

    # Copy all files to train
    print("\nCopying training files...")
    for i, (audio_file, f0_file) in enumerate(zip(audio_files, corrected_files)):
        # Copy audio
        shutil.copy(audio_file, train_dir / audio_file.name)

        # Copy f0
        shutil.copy(f0_file, train_f0_dir / f0_file.name)

        if (i + 1) % 10 == 0:
            print(f"  Copied {i + 1}/{len(audio_files)}...")

    print(f"✓ Copied {len(audio_files)} training samples")

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Training data:   {train_dir}/")
    print(f"Training f0:     {train_f0_dir}/")
    print("\nYou can now run training with:")
    print(f"  cd train/")
    print(f"  python train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare training data from corrected samples')
    parser.add_argument('--audio_dir', type=str, default='data/rumbles',
                        help='Directory with WAV files (default: data/rumbles)')
    parser.add_argument('--corrected_f0_dir', type=str, default='data/rumbles/f0_corrected',
                        help='Directory with corrected f0 CSV files (default: data/rumbles/f0_corrected)')
    parser.add_argument('--output_dir', type=str, default='train_subset',
                        help='Base output directory (default: data/rumbles)')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution for f0 directory (default: 0.016)')

    args = parser.parse_args()

    prepare_training_data(
        audio_dir=args.audio_dir,
        corrected_f0_dir=args.corrected_f0_dir,
        output_dir=args.output_dir,
        frame_resolution=args.frame_resolution
    )
