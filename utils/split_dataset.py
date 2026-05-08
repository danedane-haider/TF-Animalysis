"""
Split preprocessed dataset into train and test sets.
Keeps the f0 CSV directory in place (no splitting).
"""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random


def split_preprocessed_dataset(
    input_dir,
    test_size=0.10,
    random_state=42,
    frame_resolution=0.016
):
    """
    Split preprocessed audio files into train/test while keeping f0 data in place.

    Args:
        input_dir: Directory with *.wav and f0_corrected/*.csv
        test_size: Fraction of data for test set
        random_state: Random seed
        frame_resolution: Frame resolution (to find f0 directory)
    """
    input_dir = Path(input_dir)

    # Find audio files
    audio_files = list(input_dir.glob("*.wav"))

    if len(audio_files) == 0:
        print(f"ERROR: No .wav files found in {input_dir}")
        print("Make sure you've run preprocess_elephant.py first!")
        return

    # Check for f0 directory
    f0_dir = input_dir / "f0_corrected"
    if not f0_dir.exists():
        print(f"WARNING: F0 directory not found: {f0_dir}")
        print("Proceeding without f0 checks.")
        f0_dir = None

    print("\n" + "="*60)
    print("DATASET SPLITTING")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Audio files: {len(audio_files)}")
    print(f"F0 directory: {f0_dir}")
    print(f"Test size: {test_size*100:.1f}%")
    print(f"Random seed: {random_state}")
    print("="*60 + "\n")

    # Verify f0 files exist for all audio files (if available)
    if f0_dir is not None:
        missing_f0 = []
        for audio_file in audio_files:
            f0_file = f0_dir / f"{audio_file.stem}.f0.csv"
            if not f0_file.exists():
                missing_f0.append(audio_file.name)

        if missing_f0:
            print(f"WARNING: {len(missing_f0)} audio files missing f0 data:")
            for f in missing_f0[:5]:
                print(f"  - {f}")
            if len(missing_f0) > 5:
                print(f"  ... and {len(missing_f0)-5} more")
            print("\nRemoving files without f0 data...")
            audio_files = [f for f in audio_files if f.name not in missing_f0]
            print(f"Proceeding with {len(audio_files)} files")

    # Sort for reproducibility
    audio_files = sorted(audio_files)

    # Set random seed
    random.seed(random_state)

    # Split
    train_files, test_files = train_test_split(
        audio_files,
        test_size=test_size,
        random_state=random_state
    )

    print(f"\nSplit: {len(train_files)} train, {len(test_files)} test")

    # Create output directories
    train_dir = input_dir / 'train'
    test_dir = input_dir / 'test'

    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Copy files
    print("\nCopying training files...")
    for i, audio_file in enumerate(train_files, 1):
        # Copy audio
        shutil.copy2(audio_file, train_dir / audio_file.name)

        if i % 10 == 0 or i == len(train_files):
            print(f"  {i}/{len(train_files)}", end='\r')
    print()

    print("Copying test files...")
    for i, audio_file in enumerate(test_files, 1):
        # Copy audio
        shutil.copy2(audio_file, test_dir / audio_file.name)

        if i % 10 == 0 or i == len(test_files):
            print(f"  {i}/{len(test_files)}", end='\r')
    print()

    # Save file lists
    with open(input_dir / 'train_files.txt', 'w') as f:
        for file in sorted(train_files):
            f.write(f"{file.name}\n")

    with open(input_dir / 'test_files.txt', 'w') as f:
        for file in sorted(test_files):
            f.write(f"{file.name}\n")

    print("\n" + "="*60)
    print("SPLITTING COMPLETE")
    print("="*60)
    print(f"✓ Train: {train_dir}")
    print(f"    └── {len(train_files)} audio files")
    print(f"✓ Test:  {test_dir}")
    print(f"    └── {len(test_files)} audio files")
    print(f"\n✓ File lists saved:")
    print(f"    - {input_dir}/train_files.txt")
    print(f"    - {input_dir}/test_files.txt")

    print(f"\nDirectory structure:")
    print(f"  {input_dir}/")
    print(f"    ├── f0_corrected/  (unchanged)")
    print(f"    ├── train/")
    print(f"    │   └── *.wav")
    print(f"    ├── test/")
    print(f"    │   └── *.wav")
    print(f"    ├── train_files.txt")
    print(f"    └── test_files.txt")

    print(f"\nNext steps:")
    print(f"  1. Update configs/elephant.yaml:")
    print(f"       train: ../{input_dir}/train/")
    print(f"       test: ../{input_dir}/test/")
    print(f"  2. cd train && python train.py")


def main():
    parser = argparse.ArgumentParser(
        description='Split preprocessed dataset into train/test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python split_dataset.py \\
      --input data/rumbles \\
      --test_size 0.15 \\
      --frame_resolution 0.016

This will create:
  data/rumbles/train/  (audio files)
  data/rumbles/test/   (audio files)
The f0_corrected/ directory remains unchanged.
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with preprocessed data (*.wav + f0_corrected/*.csv)')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Fraction of data for test set (default: 0.1)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution to find f0 directory (default: 0.016)')

    args = parser.parse_args()

    split_preprocessed_dataset(
        input_dir=args.input,
        test_size=args.test_size,
        random_state=args.random_state,
        frame_resolution=args.frame_resolution
    )


if __name__ == "__main__":
    main()
