"""
Realign corrected f0 CSVs so time starts at the start marker
and ends at the end marker.
"""

import argparse
from pathlib import Path

import pandas as pd


def realign_f0_markers(input_dir, output_dir_name="f0_aligned"):
    """
    Create realigned copies of f0_corrected CSVs using start/end markers.

    Args:
        input_dir: Directory containing f0_corrected/ subdirectory
        output_dir_name: Name of the output directory under input_dir
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

    output_dir = input_dir / output_dir_name
    output_dir.mkdir(exist_ok=True)

    print(f"Input:  {f0_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(csv_files)}")

    processed = 0
    skipped = 0

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        required_cols = {"time", "start_point", "end_point"}
        if not required_cols.issubset(df.columns):
            print(f"  Skipping {csv_path.name}: missing columns {required_cols - set(df.columns)}")
            skipped += 1
            continue

        start_idx = df.index[df["start_point"] == 1]
        end_idx = df.index[df["end_point"] == 1]

        if len(start_idx) == 0 or len(end_idx) == 0:
            print(f"  Skipping {csv_path.name}: missing start or end marker")
            skipped += 1
            continue

        start_idx = int(start_idx[0])
        end_idx = int(end_idx[-1])

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        df_slice = df.loc[start_idx:end_idx].copy()
        t0 = df_slice["time"].iloc[0]
        df_slice["time"] = df_slice["time"] - t0

        df_out = df_slice.copy()
        for col in ("start_point", "end_point"):
            if col in df_out.columns:
                df_out = df_out.drop(columns=[col])

        output_path = output_dir / csv_path.name
        df_out.to_csv(output_path, index=False, float_format="%.3f")
        processed += 1

    print(f"\n✓ Realigned: {processed} files")
    if skipped:
        print(f"⊘ Skipped: {skipped} files")


def main():
    parser = argparse.ArgumentParser(
        description="Realign f0_corrected CSVs to start at start_marker and end at end_marker"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing f0_corrected/ subdirectory",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="f0_aligned",
        help="Output directory name under input_dir (default: f0_aligned)",
    )

    args = parser.parse_args()

    realign_f0_markers(
        input_dir=args.input,
        output_dir_name=args.output_dir_name,
    )


if __name__ == "__main__":
    main()
