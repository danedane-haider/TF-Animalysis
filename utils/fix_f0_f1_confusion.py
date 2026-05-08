"""
Fix F0-F1 confusion by multiplying F0 values by 2 for specific file ranges.

When F0 extraction mistakenly tracked F1 instead of F0, the true F0 is approximately
half of the detected value. Multiplying by 2 corrects this.
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re


def is_in_ranges(num, ranges):
    """Check if number is in any of the given ranges (inclusive)."""
    for start, end in ranges:
        if start <= num <= end:
            return True
    return False


def fix_f0_f1_confusion(input_dir, ranges, dry_run=False, create_backup=True):
    """
    Multiply F0 values by 2 for files in specified ranges.

    Args:
        input_dir: Directory containing f0_corrected CSV files
        ranges: List of (start, end) tuples for file number ranges
        dry_run: If True, only print what would be changed without modifying files
        create_backup: If True, create backup before modifying
    """
    input_dir = Path(input_dir)
    f0_corrected_dir = input_dir / "f0_corrected"

    if not f0_corrected_dir.exists():
        raise ValueError(f"f0_corrected directory not found: {f0_corrected_dir}")

    # Find all CSV files and sort them
    csv_files = sorted(list(f0_corrected_dir.glob("*.f0.csv")))

    if len(csv_files) == 0:
        raise ValueError(f"No .f0.csv files found in {f0_corrected_dir}")

    # Pre-scan to count how many will be modified
    # file_num is the 1-indexed position in the sorted list
    files_to_modify = []
    for idx, csv_path in enumerate(csv_files, start=1):
        if is_in_ranges(idx, ranges):
            files_to_modify.append((csv_path, idx))

    print("="*60)
    print("FIX F0-F1 CONFUSION (MULTIPLY F0 BY 2)")
    print("="*60)
    print(f"Directory: {f0_corrected_dir}")
    print(f"Total files in directory: {len(csv_files)}")
    print(f"Files that will be modified: {len(files_to_modify)}")
    print(f"Files that will be skipped: {len(csv_files) - len(files_to_modify)}")
    print(f"\nRanges to fix:")
    for start, end in ranges:
        count = sum(1 for _, num in files_to_modify if start <= num <= end)
        print(f"  {start:4d}-{end:4d} ({count} files)")
    print(f"\nDry run: {dry_run}")
    if not dry_run and create_backup:
        print(f"Backup: will create .bak files")
    print()

    if len(files_to_modify) == 0:
        print("No files to modify!")
        return

    # Process files
    modified_count = 0

    for csv_path, file_num in tqdm(files_to_modify, desc="Processing"):

        # Read CSV
        df = pd.read_csv(csv_path)

        if 'frequency' not in df.columns:
            print(f"  Warning: 'frequency' column not found in {csv_path.name}, skipping")
            skipped_count += 1
            continue

        # Get original values for reporting
        orig_mean = df['frequency'][df['frequency'] > 0].mean() if (df['frequency'] > 0).any() else 0

        if not dry_run:
            # Multiply frequency by 2
            df['frequency'] = df['frequency'] * 2.0

            # Save back to CSV
            df.to_csv(csv_path, index=False)

        new_mean = orig_mean * 2.0

        if dry_run:
            print(f"  [{file_num:04d}] {csv_path.name}: {orig_mean:.1f} Hz -> {new_mean:.1f} Hz (DRY RUN)")

        modified_count += 1

    print()
    print("="*60)
    print("✓ Complete!")
    print("="*60)
    print(f"Modified: {modified_count} files")
    print(f"Skipped: {len(csv_files) - len(files_to_modify)} files")

    if dry_run:
        print()
        print("This was a DRY RUN. No files were actually modified.")
        print("Run without --dry_run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description='Fix F0-F1 confusion by multiplying F0 by 2 for specific file ranges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Dry run (preview changes)
  python preprocessing/fix_f0_f1_confusion.py --input data/ntvow --dry_run

  # Apply changes
  python preprocessing/fix_f0_f1_confusion.py --input data/ntvow

File ranges are specified in the script. Edit the script to change ranges.
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (should contain f0_corrected/ subdirectory)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview changes without modifying files')

    args = parser.parse_args()

    # Define ranges (inclusive)
    ranges = [
        (2955, 3190)
    ]

    fix_f0_f1_confusion(
        input_dir=args.input,
        ranges=ranges,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
