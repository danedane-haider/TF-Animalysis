"""
Refine f0_corrected contours via multi-scale harmonic salience (MSS-style).

For each frame, the script searches neighboring base-frequency bins around the
current f0 and selects the candidate with the highest summed harmonic energy
across multiple STFT sizes.

Note:
Input CSV `frequency` values are treated as first-harmonic frequencies.
Internally, refinement is run in fundamental-frequency space (divide by 2),
and the refined result is mapped back to first-harmonic frequency (multiply by 2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def _nearest_bin_indices(freqs: np.ndarray, target_freqs: np.ndarray) -> np.ndarray:
    """Return nearest frequency-bin index for each target frequency."""
    idx = np.searchsorted(freqs, target_freqs)
    idx = np.clip(idx, 1, len(freqs) - 1)
    left = freqs[idx - 1]
    right = freqs[idx]
    choose_right = (target_freqs - left) > (right - target_freqs)
    return idx - 1 + choose_right.astype(np.int64)


def _nearest_bin_index_scalar(freqs: np.ndarray, target_freq: float) -> int:
    """Scalar nearest-bin helper."""
    idx = int(np.searchsorted(freqs, target_freq))
    idx = int(np.clip(idx, 1, len(freqs) - 1))
    if (target_freq - freqs[idx - 1]) <= (freqs[idx] - target_freq):
        return idx - 1
    return idx


def _smooth_voiced_median(f0_hz: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Median-smooth only voiced regions (>0), preserving unvoiced zeros.
    """
    if kernel_size <= 1:
        return f0_hz
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    out = f0_hz.copy()
    voiced = f0_hz > 0.0
    if not np.any(voiced):
        return out

    starts = np.where(np.diff(np.concatenate(([0], voiced.astype(np.int8), [0]))) == 1)[0]
    ends = np.where(np.diff(np.concatenate(([0], voiced.astype(np.int8), [0]))) == -1)[0]

    for s, e in zip(starts, ends):
        segment = out[s:e]
        if segment.size == 0:
            continue
        padded = np.pad(segment, (pad, pad), mode="edge")
        smoothed = np.empty_like(segment)
        for i in range(segment.size):
            smoothed[i] = np.median(padded[i : i + kernel_size])
        out[s:e] = smoothed

    return out


def _quadratic_peak_offset(y_left: float, y_center: float, y_right: float) -> float:
    """
    Sub-bin offset from a 3-point quadratic fit around a local maximum.
    Returns an offset in bins, typically within [-1, 1].
    """
    denom = (y_left - (2.0 * y_center) + y_right)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (y_left - y_right) / denom
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -1.0, 1.0))


def _compute_scale_power(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    power_exp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrogram and frequency bins for one STFT scale."""
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
    stft = librosa.stft(
        y=audio,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** power_exp
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return power, freqs


def refine_f0_file(
    audio_path: Path,
    f0_csv_path: Path,
    output_csv_path: Path,
    sr: int,
    n_fft: int,
    win_lengths: list[int],
    hop_length: int,
    n_harmonics: int,
    neighbor_radius_bins: int,
    harmonic_bin_radius: int,
    power_exp: float,
    harmonic_weight_exp: float,
    min_f0_hz: float,
    median_kernel: int,
    subbin_interpolation: bool,
) -> tuple[bool, float]:
    """
    Refine one f0 CSV. Returns (success, mean_abs_delta_hz).
    """
    df = pd.read_csv(f0_csv_path)
    required_cols = {"time", "frequency"}
    if not required_cols.issubset(df.columns):
        print(f"  Skipping {f0_csv_path.name}: missing columns {required_cols - set(df.columns)}")
        return False, 0.0

    if not audio_path.exists():
        print(f"  Skipping {f0_csv_path.name}: audio not found ({audio_path.name})")
        return False, 0.0

    audio, _ = librosa.load(audio_path, sr=sr)
    times = df["time"].to_numpy(dtype=float)
    # The stored values are first-harmonic frequencies. Refine in F0 space.
    f0_in_h1 = df["frequency"].to_numpy(dtype=float)
    f0_in = 0.5 * f0_in_h1

    scale_data = []
    for win_length in win_lengths:
        power, freqs = _compute_scale_power(
            audio=audio,
            sr=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power_exp=power_exp,
        )
        frame_idx = np.rint(times * sr / hop_length).astype(np.int64)
        frame_idx = np.clip(frame_idx, 0, max(power.shape[1] - 1, 0))
        scale_data.append((win_length, power, freqs, frame_idx))

    if len(scale_data) == 0:
        return False, 0.0

    _, _, ref_freqs, _ = scale_data[0]
    ref_bin_positions = np.arange(len(ref_freqs), dtype=np.float64)
    nyquist = float(sr) * 0.5

    f0_out = f0_in.copy()
    offsets = np.arange(-neighbor_radius_bins, neighbor_radius_bins + 1, dtype=np.int64)

    for i in range(len(f0_in)):
        f0_now = float(f0_in[i])
        if not np.isfinite(f0_now) or f0_now <= min_f0_hz:
            f0_out[i] = 0.0 if f0_now <= 0.0 else f0_now
            continue

        base_bin = _nearest_bin_index_scalar(ref_freqs, f0_now)
        cand_bins = base_bin + offsets
        cand_bins = cand_bins[(cand_bins >= 1) & (cand_bins < len(ref_freqs))]
        if cand_bins.size == 0:
            continue

        cand_scores = []
        valid_bins = []

        for cand_bin in cand_bins:
            cand_f0 = float(ref_freqs[int(cand_bin)])
            if cand_f0 <= min_f0_hz:
                continue

            max_h = min(n_harmonics, int(nyquist // max(cand_f0, 1e-8)))
            if max_h < 1:
                continue

            score = 0.0
            for _, power, freqs, frame_idx in scale_data:
                frame = int(frame_idx[i])
                if frame < 0 or frame >= power.shape[1]:
                    continue

                spec_frame = power[:, frame]
                norm = float(np.mean(spec_frame)) + 1e-12
                for h in range(1, max_h + 1):
                    target_hz = float(h) * cand_f0
                    if target_hz >= nyquist:
                        break
                    h_bin = _nearest_bin_index_scalar(freqs, target_hz)
                    lo = max(0, h_bin - harmonic_bin_radius)
                    hi = min(len(spec_frame) - 1, h_bin + harmonic_bin_radius)
                    band_val = float(np.mean(spec_frame[lo : hi + 1]))
                    h_weight = 1.0 / (float(h) ** harmonic_weight_exp)
                    score += h_weight * (band_val / norm)

            cand_scores.append(score)
            valid_bins.append(int(cand_bin))

        if len(cand_scores) == 0:
            continue

        scores = np.asarray(cand_scores, dtype=np.float64)
        bins = np.asarray(valid_bins, dtype=np.float64)
        best_idx = int(np.argmax(scores))
        best_bin = float(bins[best_idx])

        if subbin_interpolation and 0 < best_idx < (len(scores) - 1):
            delta = _quadratic_peak_offset(
                y_left=float(scores[best_idx - 1]),
                y_center=float(scores[best_idx]),
                y_right=float(scores[best_idx + 1]),
            )
            best_bin += delta
            best_bin = float(np.clip(best_bin, 1.0, float(len(ref_freqs) - 1)))

        f0_out[i] = float(np.interp(best_bin, ref_bin_positions, ref_freqs))

    if median_kernel > 1:
        f0_out = _smooth_voiced_median(f0_out, median_kernel)

    # Map back to first-harmonic frequency for output/comparison.
    f0_out_h1 = 2.0 * f0_out
    delta = np.abs(f0_out_h1 - f0_in_h1)
    mean_abs_delta = float(np.mean(delta[np.isfinite(delta)])) if np.any(np.isfinite(delta)) else 0.0

    df_out = df.copy()
    df_out["frequency_original"] = f0_in_h1
    df_out["frequency"] = f0_out_h1
    df_out.to_csv(output_csv_path, index=False, float_format="%.6f")
    return True, mean_abs_delta


def refine_f0_folder(
    input_dir: Path,
    f0_input_dir_name: str,
    output_dir_name: str,
    sr: int,
    n_fft: int,
    win_lengths: list[int],
    hop_length: int,
    n_harmonics: int,
    neighbor_radius_bins: int,
    harmonic_bin_radius: int,
    power_exp: float,
    harmonic_weight_exp: float,
    min_f0_hz: float,
    median_kernel: int,
    subbin_interpolation: bool,
) -> None:
    """Run refinement for all .f0.csv files in input_dir/<f0_input_dir_name>."""
    f0_dir = input_dir / f0_input_dir_name
    if not f0_dir.exists():
        raise FileNotFoundError(f"F0 input directory not found: {f0_dir}")

    csv_files = sorted(f0_dir.glob("*.f0.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No .f0.csv files found in {f0_dir}")

    output_dir = input_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input dir:  {input_dir}")
    print(f"F0 input:   {f0_dir}")
    print(f"F0 output:  {output_dir}")
    print(f"Files:      {len(csv_files)}")
    print(f"STFT n_fft: {n_fft}")
    print(f"STFT win_lengths: {win_lengths}")
    print(
        "Params: "
        f"sr={sr}, hop={hop_length}, harmonics={n_harmonics}, "
        f"neighbor_radius_bins={neighbor_radius_bins}, harmonic_bin_radius={harmonic_bin_radius}, "
        f"power_exp={power_exp}, harmonic_weight_exp={harmonic_weight_exp}, "
        f"min_f0_hz={min_f0_hz}, median_kernel={median_kernel}, "
        f"subbin_interpolation={subbin_interpolation}"
    )

    processed = 0
    skipped = 0
    delta_values = []

    for csv_path in tqdm(csv_files, desc="Refining f0 (harmonic MSS)"):
        stem = csv_path.stem.replace(".f0", "")
        audio_path = input_dir / f"{stem}.wav"
        output_csv_path = output_dir / csv_path.name

        ok, mean_abs_delta = refine_f0_file(
            audio_path=audio_path,
            f0_csv_path=csv_path,
            output_csv_path=output_csv_path,
            sr=sr,
            n_fft=n_fft,
            win_lengths=win_lengths,
            hop_length=hop_length,
            n_harmonics=n_harmonics,
            neighbor_radius_bins=neighbor_radius_bins,
            harmonic_bin_radius=harmonic_bin_radius,
            power_exp=power_exp,
            harmonic_weight_exp=harmonic_weight_exp,
            min_f0_hz=min_f0_hz,
            median_kernel=median_kernel,
            subbin_interpolation=subbin_interpolation,
        )
        if ok:
            processed += 1
            delta_values.append(mean_abs_delta)
        else:
            skipped += 1

    print(f"\n✓ Refined: {processed} files")
    if skipped:
        print(f"⊘ Skipped: {skipped} files")
    if len(delta_values) > 0:
        print(f"Mean |Δf0| over files: {float(np.mean(delta_values)):.4f} Hz")


def _parse_win_lengths(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("win_lengths cannot be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refine f0_corrected contours by selecting the neighboring-bin candidate "
            "with highest multi-scale harmonic salience."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing wav files and f0_corrected/.",
    )
    parser.add_argument(
        "--f0_input_dir_name",
        type=str,
        default="f0_corrected",
        help="Base F0 input directory under input dir (default: f0_corrected).",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="f0_refined",
        help="Output directory under input dir (default: f0_refined).",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument(
        "--n_fft",
        type=int,
        default=16384,
        help="Shared FFT size across scales (default: 16384).",
    )
    parser.add_argument(
        "--win_lengths",
        type=str,
        default="4096,8192,16384",
        help="Comma-separated STFT window lengths (default: 4096,8192,16384).",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=320,
        help="Hop length in samples (default: 320 for 20ms at 16kHz).",
    )
    parser.add_argument(
        "--n_harmonics",
        type=int,
        default=10,
        help="Number of harmonics in score (default: 10).",
    )
    parser.add_argument(
        "--neighbor_radius_bins",
        type=int,
        default=3,
        help="Base-bin search radius around current f0 (default: 3).",
    )
    parser.add_argument(
        "--harmonic_bin_radius",
        type=int,
        default=0,
        help="Per-harmonic bin averaging radius (default: 0 = exact bin).",
    )
    parser.add_argument(
        "--power_exp",
        type=float,
        default=2.0,
        help="Spectral magnitude exponent (default: 2.0 for power).",
    )
    parser.add_argument(
        "--harmonic_weight_exp",
        type=float,
        default=0.5,
        help="Harmonic weight exponent: weight=1/h^exp (default: 0.5).",
    )
    parser.add_argument(
        "--min_f0_hz",
        type=float,
        default=1.0,
        help="Minimum voiced f0 in Hz (default: 1.0).",
    )
    parser.add_argument(
        "--median_kernel",
        type=int,
        default=3,
        help="Median kernel on voiced segments (default: 3, set 1 to disable).",
    )
    parser.add_argument(
        "--subbin_interpolation",
        action="store_true",
        help="Enable quadratic sub-bin interpolation around the best candidate bin.",
    )

    args = parser.parse_args()
    win_lengths = _parse_win_lengths(args.win_lengths)

    refine_f0_folder(
        input_dir=Path(args.input),
        f0_input_dir_name=args.f0_input_dir_name,
        output_dir_name=args.output_dir_name,
        sr=args.sr,
        n_fft=args.n_fft,
        win_lengths=win_lengths,
        hop_length=args.hop_length,
        n_harmonics=args.n_harmonics,
        neighbor_radius_bins=args.neighbor_radius_bins,
        harmonic_bin_radius=args.harmonic_bin_radius,
        power_exp=args.power_exp,
        harmonic_weight_exp=args.harmonic_weight_exp,
        min_f0_hz=args.min_f0_hz,
        median_kernel=args.median_kernel,
        subbin_interpolation=args.subbin_interpolation,
    )


if __name__ == "__main__":
    main()
