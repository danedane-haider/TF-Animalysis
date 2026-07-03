"""Extract Elelet amplitudes along an existing contour.

The script reads contour CSV files such as ``f0_refined/*.f0.csv`` and
samples the matching precomputed Elelet representation at each contour point.
It adds an amplitude column to each CSV, writing in place by default or into a
separate output directory when requested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f0_extraction.extract_f0 import resolve_precomputed_dir
from f0_extraction.pipeline import representation_dir


def _resolve_child_or_absolute(base_dir: Path, path_name: str | Path) -> Path:
    path = Path(path_name)
    if path.is_absolute():
        return path
    return base_dir / path


def _contour_stem(csv_path: Path) -> str:
    suffix = ".f0.csv"
    if csv_path.name.endswith(suffix):
        return csv_path.name[: -len(suffix)]
    return csv_path.stem


def _nearest_bin_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return nearest index in ``values`` for each target value."""
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1D array")
    if len(values) == 1:
        return np.zeros(len(targets), dtype=int)

    order = np.argsort(values)
    sorted_values = values[order]
    idx = np.searchsorted(sorted_values, targets)
    idx = np.clip(idx, 1, len(sorted_values) - 1)

    left_idx = idx - 1
    right_idx = idx
    left = sorted_values[left_idx]
    right = sorted_values[right_idx]
    choose_right = targets - left > right - targets
    sorted_nearest = left_idx + choose_right.astype(int)
    return order[sorted_nearest]


def _load_elelet_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Elelet amplitude grid, times, and channel-center frequencies."""
    with np.load(npz_path) as data:
        if "coeffs_abs" in data:
            amplitudes = np.asarray(data["coeffs_abs"], dtype=np.float64)
        elif "coeffs" in data:
            amplitudes = np.abs(data["coeffs"]).astype(np.float64)
        else:
            raise ValueError(f"{npz_path} does not contain coeffs_abs or coeffs")

        if "fc" not in data:
            raise ValueError(f"{npz_path} does not contain Elelet channel frequencies (fc)")
        freqs = np.asarray(data["fc"], dtype=np.float64)

        if "times" in data:
            times = np.asarray(data["times"], dtype=np.float64)
        elif {"sr", "stride"}.issubset(data.files):
            sr = float(np.asarray(data["sr"]).item())
            stride = float(np.asarray(data["stride"]).item())
            times = np.arange(amplitudes.shape[1], dtype=np.float64) * stride / sr
        else:
            raise ValueError(f"{npz_path} does not contain times or sr/stride metadata")

    if amplitudes.ndim != 2:
        raise ValueError(f"Elelet amplitude grid must be 2D, got shape {amplitudes.shape}")
    if len(freqs) != amplitudes.shape[0]:
        raise ValueError(
            f"Frequency axis length {len(freqs)} does not match amplitude rows {amplitudes.shape[0]}"
        )
    if len(times) != amplitudes.shape[1]:
        raise ValueError(
            f"Time axis length {len(times)} does not match amplitude columns {amplitudes.shape[1]}"
        )

    return amplitudes, times, freqs


def extract_elelet_amplitude_for_contour(
    contour_csv_path: str | Path,
    elelet_npz_path: str | Path,
    output_csv_path: str | Path | None = None,
    *,
    time_column: str = "time",
    frequency_column: str = "frequency",
    amplitude_column: str = "elelet_amplitude",
    frequency_multiplier: float = 1.0,
    fill_value: float = 0.0,
    include_indices: bool = False,
) -> tuple[int, int]:
    """Extract Elelet amplitudes for one contour CSV.

    Returns:
        A ``(valid_points, total_points)`` tuple.
    """
    contour_csv_path = Path(contour_csv_path)
    elelet_npz_path = Path(elelet_npz_path)
    output_csv_path = Path(output_csv_path) if output_csv_path is not None else contour_csv_path

    df = pd.read_csv(contour_csv_path)
    required_columns = {time_column, frequency_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{contour_csv_path} is missing required columns: {missing}")

    elelet_amplitudes, elelet_times, elelet_freqs = _load_elelet_npz(elelet_npz_path)
    contour_times = df[time_column].to_numpy(dtype=np.float64)
    contour_freqs = df[frequency_column].to_numpy(dtype=np.float64) * frequency_multiplier

    valid = (
        np.isfinite(contour_times)
        & np.isfinite(contour_freqs)
        & (contour_freqs > 0)
        & (contour_times >= float(np.min(elelet_times)))
        & (contour_times <= float(np.max(elelet_times)))
        & (contour_freqs >= float(np.min(elelet_freqs)))
        & (contour_freqs <= float(np.max(elelet_freqs)))
    )

    amplitudes = np.full(len(df), fill_value, dtype=np.float64)
    frame_indices = np.full(len(df), -1, dtype=int)
    channel_indices = np.full(len(df), -1, dtype=int)

    if np.any(valid):
        valid_indices = np.where(valid)[0]
        frame_indices[valid] = _nearest_bin_indices(elelet_times, contour_times[valid])
        channel_indices[valid] = _nearest_bin_indices(elelet_freqs, contour_freqs[valid])
        amplitudes[valid] = elelet_amplitudes[channel_indices[valid], frame_indices[valid]]
    else:
        valid_indices = np.array([], dtype=int)

    df[amplitude_column] = amplitudes

    if include_indices:
        channel_frequency = np.full(len(df), fill_value, dtype=np.float64)
        frame_time = np.full(len(df), fill_value, dtype=np.float64)
        if len(valid_indices):
            channel_frequency[valid] = elelet_freqs[channel_indices[valid]]
            frame_time[valid] = elelet_times[frame_indices[valid]]
        df[f"{amplitude_column}_frame"] = frame_indices
        df[f"{amplitude_column}_channel"] = channel_indices
        df[f"{amplitude_column}_time"] = frame_time
        df[f"{amplitude_column}_frequency"] = channel_frequency

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, float_format="%.6f")
    return int(np.count_nonzero(valid)), int(len(df))


def extract_elelet_amplitudes(
    input_dir: str | Path,
    contour_dir_name: str = "f0_refined",
    elelet_dir_name: str | None = None,
    output_dir_name: str | None = None,
    *,
    use_precomputed_representations: str | Path | None = None,
    representation_fmax: float = 750.0,
    time_column: str = "time",
    frequency_column: str = "frequency",
    amplitude_column: str = "elelet_amplitude",
    frequency_multiplier: float = 1.0,
    fill_value: float = 0.0,
    include_indices: bool = False,
) -> None:
    """Extract Elelet amplitudes for every contour CSV in a dataset."""
    input_dir = Path(input_dir)
    contour_dir = _resolve_child_or_absolute(input_dir, contour_dir_name)
    if use_precomputed_representations is not None:
        elelet_dir = resolve_precomputed_dir(use_precomputed_representations, "elelet")
    else:
        elelet_dir = _resolve_child_or_absolute(
            input_dir,
            elelet_dir_name or representation_dir("elelet", representation_fmax),
        )
    output_dir = contour_dir if output_dir_name is None else _resolve_child_or_absolute(input_dir, output_dir_name)

    if not contour_dir.exists():
        print(f"ERROR: contour directory not found: {contour_dir}")
        return
    if not elelet_dir.exists():
        print(f"ERROR: Elelet representation directory not found: {elelet_dir}")
        return

    csv_files = sorted(contour_dir.glob("*.f0.csv"))
    if not csv_files:
        print(f"ERROR: no .f0.csv files found in {contour_dir}")
        return

    print("=" * 60)
    print("ELELET CONTOUR AMPLITUDE EXTRACTION")
    print("=" * 60)
    print(f"Input:       {input_dir}")
    print(f"Contours:    {contour_dir}")
    print(f"Elelet:      {elelet_dir}")
    print(f"Output:      {output_dir if output_dir != contour_dir else 'in-place'}")
    print(f"Files:       {len(csv_files)}")
    print(f"Columns:     time={time_column}, frequency={frequency_column}, amplitude={amplitude_column}")
    if frequency_multiplier != 1.0:
        print(f"Lookup freq: {frequency_column} * {frequency_multiplier:g}")
    print("=" * 60 + "\n")

    processed = 0
    skipped = 0
    valid_points = 0
    total_points = 0

    for csv_path in tqdm(csv_files, desc="Extracting Elelet amplitudes"):
        stem = _contour_stem(csv_path)
        elelet_path = elelet_dir / f"{stem}.npz"
        if not elelet_path.exists():
            print(f"  Skipping {csv_path.name}: Elelet file not found ({elelet_path.name})")
            skipped += 1
            continue

        output_path = output_dir / csv_path.name
        try:
            valid, total = extract_elelet_amplitude_for_contour(
                csv_path,
                elelet_path,
                output_path,
                time_column=time_column,
                frequency_column=frequency_column,
                amplitude_column=amplitude_column,
                frequency_multiplier=frequency_multiplier,
                fill_value=fill_value,
                include_indices=include_indices,
            )
        except Exception as exc:
            print(f"  Skipping {csv_path.name}: {exc}")
            skipped += 1
            continue

        processed += 1
        valid_points += valid
        total_points += total

    print("\n" + "=" * 60)
    print("ELELET AMPLITUDE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed}/{len(csv_files)} files")
    if skipped:
        print(f"Skipped:   {skipped} files")
    print(f"Samples:   {valid_points}/{total_points} contour points inside Elelet grid")
    print(f"Output:    {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Elelet amplitudes at an existing contour.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Directory containing audio, contours, and Elelet .npz files.")
    parser.add_argument("--contour_dir", type=str, default="f0_refined", help="Contour directory under input, or absolute path.")
    parser.add_argument("--use_precomputed_representations", type=str, default=None, help="Path to a precomputed Elelet folder, or a parent containing elelet_*.")
    parser.add_argument("--elelet_dir", type=str, default=None, help="Legacy alias for --use_precomputed_representations.")
    parser.add_argument("--output_dir_name", type=str, default=None, help="Output directory under input; default writes in-place.")
    parser.add_argument("--representation_fmax", type=float, default=750.0, help="Fmax label used for the default elelet_<fmax> directory.")
    parser.add_argument("--time_column", type=str, default="time", help="Contour time column.")
    parser.add_argument("--frequency_column", type=str, default="frequency", help="Contour frequency column.")
    parser.add_argument("--amplitude_column", type=str, default="elelet_amplitude", help="Output amplitude column.")
    parser.add_argument("--frequency_multiplier", type=float, default=1.0, help="Multiply contour frequencies before Elelet lookup.")
    parser.add_argument("--fill_value", type=float, default=0.0, help="Value for invalid/out-of-grid contour points.")
    parser.add_argument("--include_indices", action="store_true", help="Also write Elelet frame/channel lookup columns.")
    args = parser.parse_args()

    extract_elelet_amplitudes(
        input_dir=args.input,
        contour_dir_name=args.contour_dir,
        elelet_dir_name=args.elelet_dir,
        output_dir_name=args.output_dir_name,
        use_precomputed_representations=args.use_precomputed_representations,
        representation_fmax=args.representation_fmax,
        time_column=args.time_column,
        frequency_column=args.frequency_column,
        amplitude_column=args.amplitude_column,
        frequency_multiplier=args.frequency_multiplier,
        fill_value=args.fill_value,
        include_indices=args.include_indices,
    )


if __name__ == "__main__":
    main()
