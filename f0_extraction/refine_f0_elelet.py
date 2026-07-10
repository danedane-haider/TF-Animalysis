"""
Refine first-harmonic contours via Elelet-based harmonic salience.

For each frame, the script searches neighboring candidate bins around the
current base frequency in auditory-scale space and selects the candidate with
the highest summed harmonic energy in Elelet coefficients.

Input note:
`frequency` in the input CSV is treated as first-harmonic frequency (H1).
Internally we refine in F0 space:
  F0 = H1 / 2
and map back for output:
  H1_refined = 2 * F0_refined
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f0_extraction.extract_f0 import (
    load_precomputed_representation,
    read_metainfo,
    resolve_precomputed_dir,
)


def _to_numpy(x: torch.Tensor | np.ndarray | float) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _freq_to_aud(freq_hz: np.ndarray | float, scale: str, fs: int) -> np.ndarray:
    from tf_representations.utils_auditory_scales import freqtoaud

    freq_t = torch.as_tensor(freq_hz, dtype=torch.float64)
    return _to_numpy(freqtoaud(freq_t, scale=scale, fs=fs)).astype(np.float64)


def _aud_to_freq(aud: np.ndarray | float, scale: str, fs: int) -> np.ndarray:
    from tf_representations.utils_auditory_scales import audtofreq

    aud_t = torch.as_tensor(aud, dtype=torch.float64)
    return _to_numpy(audtofreq(aud_t, scale=scale, fs=fs)).astype(np.float64)


def _nearest_sorted_idx(sorted_values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(sorted_values, target))
    idx = int(np.clip(idx, 1, len(sorted_values) - 1))
    left = sorted_values[idx - 1]
    right = sorted_values[idx]
    if (target - left) <= (right - target):
        return idx - 1
    return idx


def _nearest_sorted_indices(sorted_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_values, targets)
    idx = np.clip(idx, 1, len(sorted_values) - 1)
    left = sorted_values[idx - 1]
    right = sorted_values[idx]
    choose_right = (targets - left) > (right - targets)
    return idx - 1 + choose_right.astype(np.int64)


def _quadratic_peak_offset(y_left: float, y_center: float, y_right: float) -> float:
    denom = (y_left - (2.0 * y_center) + y_right)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (y_left - y_right) / denom
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -1.0, 1.0))


def _smooth_voiced_median(f0_hz: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return f0_hz
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    out = f0_hz.copy()
    voiced = f0_hz > 0.0
    if not np.any(voiced):
        return out

    edges = np.diff(np.concatenate(([0], voiced.astype(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    for s, e in zip(starts, ends):
        seg = out[s:e]
        if seg.size == 0:
            continue
        padded = np.pad(seg, (pad, pad), mode="edge")
        smoothed = np.empty_like(seg)
        for i in range(seg.size):
            smoothed[i] = np.median(padded[i : i + kernel_size])
        out[s:e] = smoothed

    return out


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("kernel_sizes cannot be empty")
    return values


def _compute_elelet_power(
    audio: np.ndarray,
    sr: int,
    kernel_size: int,
    num_channels: int,
    stride: int,
    f_min: float,
    f_max: float,
    supp_mult: float,
    scale: str,
    power_exp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from tf_representations.transforms import Elelet

    transform = Elelet(
        kernel_size=kernel_size,
        num_channels=num_channels,
        stride=stride,
        f_min=f_min,
        f_max=f_max,
        fs=sr,
        supp_mult=supp_mult,
        scale=scale,
        use_torch=False,
    )
    coeffs = transform(audio)
    power = np.abs(coeffs) ** power_exp

    fc_hz = np.asarray(transform.fc, dtype=np.float64)
    n_channels = power.shape[0]
    fc_hz = fc_hz[:n_channels]
    power = power[: len(fc_hz), :]

    valid = (fc_hz >= f_min) & (fc_hz <= f_max)
    if np.any(valid):
        fc_hz = fc_hz[valid]
        power = power[valid, :]

    fc_aud = _freq_to_aud(fc_hz, scale=scale, fs=sr)
    order = np.argsort(fc_aud)
    return power[order, :], fc_hz[order], fc_aud[order]


def _load_precomputed_elelet_power(
    audio_path: Path,
    precomputed_dir: Path,
    times: np.ndarray,
    sr: int,
    f_min: float,
    f_max: float,
    scale: str,
    power_exp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    magnitude, fc_hz, repr_times = load_precomputed_representation(precomputed_dir, audio_path, "elelet")
    power = magnitude ** power_exp

    valid = (fc_hz >= f_min) & (fc_hz <= f_max)
    if np.any(valid):
        fc_hz = fc_hz[valid]
        power = power[valid, :]

    fc_aud = _freq_to_aud(fc_hz, scale=scale, fs=sr)
    order = np.argsort(fc_aud)
    frame_idx = _nearest_sorted_indices(repr_times, times)
    frame_idx = np.clip(frame_idx, 0, power.shape[1] - 1)

    meta = read_metainfo(precomputed_dir)
    params = meta.get("parameters", {})
    kernel_size = int(params.get("kernel_size", 0))
    stride = int(params.get("stride", 0))
    if stride <= 0 and len(repr_times) > 1:
        stride = max(1, int(round(float(np.median(np.diff(repr_times))) * sr)))

    return power[order, :], fc_hz[order], fc_aud[order], frame_idx, kernel_size, stride


def refine_f0_file(
    audio_path: Path,
    f0_csv_path: Path,
    output_csv_path: Path,
    sr: int,
    kernel_size: int,
    kernel_sizes: list[int] | None,
    num_channels: int,
    stride: int,
    elelet_fmin: float,
    elelet_fmax: float,
    supp_mult: float,
    scale: str,
    n_harmonics: int,
    neighbor_radius_bins: int,
    harmonic_bin_radius: int,
    power_exp: float,
    harmonic_weight_exp: float,
    min_f0_hz: float,
    median_kernel: int,
    subbin_interpolation: bool,
    precomputed_dir: Path | None = None,
) -> tuple[bool, float]:
    df = pd.read_csv(f0_csv_path)
    required_cols = {"time", "frequency"}
    if not required_cols.issubset(df.columns):
        print(f"  Skipping {f0_csv_path.name}: missing columns {required_cols - set(df.columns)}")
        return False, 0.0
    if precomputed_dir is None and not audio_path.exists():
        print(f"  Skipping {f0_csv_path.name}: audio not found ({audio_path.name})")
        return False, 0.0

    times = df["time"].to_numpy(dtype=float)
    f0_in_h1 = df["frequency"].to_numpy(dtype=float)
    f0_in = 0.5 * f0_in_h1

    kernel_sizes = kernel_sizes if kernel_sizes is not None and len(kernel_sizes) > 0 else [kernel_size]

    scale_data = []
    edge_kernel_sizes = list(kernel_sizes)
    edge_stride = stride
    if precomputed_dir is not None:
        try:
            power, fc_hz, fc_aud, frame_idx, precomputed_kernel_size, precomputed_stride = _load_precomputed_elelet_power(
                audio_path=audio_path,
                precomputed_dir=precomputed_dir,
                times=times,
                sr=sr,
                f_min=elelet_fmin,
                f_max=elelet_fmax,
                scale=scale,
                power_exp=power_exp,
            )
        except Exception as exc:
            print(f"  Skipping {f0_csv_path.name}: {exc}")
            return False, 0.0
        if power.shape[1] > 0 and power.shape[0] >= 3:
            scale_data.append((precomputed_kernel_size or kernel_size, power, fc_hz, fc_aud, frame_idx))
            edge_kernel_sizes = [precomputed_kernel_size or kernel_size]
            edge_stride = precomputed_stride or stride
    else:
        audio, _ = librosa.load(audio_path, sr=sr)
        for ks in kernel_sizes:
            power, fc_hz, fc_aud = _compute_elelet_power(
                audio=audio,
                sr=sr,
                kernel_size=ks,
                num_channels=num_channels,
                stride=stride,
                f_min=elelet_fmin,
                f_max=elelet_fmax,
                supp_mult=supp_mult,
                scale=scale,
                power_exp=power_exp,
            )
            if power.shape[1] == 0 or power.shape[0] < 3:
                continue
            frame_idx = np.rint(times * sr / stride).astype(np.int64)
            frame_idx = np.clip(frame_idx, 0, power.shape[1] - 1)
            scale_data.append((int(ks), power, fc_hz, fc_aud, frame_idx))

    if len(scale_data) == 0:
        print(f"  Skipping {f0_csv_path.name}: invalid Elelet representation")
        return False, 0.0

    _, _, _, ref_fc_aud, _ = scale_data[0]
    channel_pos = np.arange(len(ref_fc_aud), dtype=np.float64)
    nyquist = 0.5 * float(sr)
    max_kernel_size = int(max(edge_kernel_sizes))
    edge_frames = int(max_kernel_size // 2 // max(edge_stride, 1))

    f0_out = f0_in.copy()
    offsets_cache: dict[int, np.ndarray] = {}
    n_frames = len(f0_in)

    for i in range(n_frames):
        f0_now = float(f0_in[i])
        if not np.isfinite(f0_now) or f0_now <= min_f0_hz:
            f0_out[i] = 0.0 if f0_now <= 0.0 else f0_now
            continue

        if edge_frames > 0 and (i < edge_frames or i >= (n_frames - edge_frames)):
            local_neighbor_radius = 1
        else:
            local_neighbor_radius = neighbor_radius_bins
        local_neighbor_radius = max(0, int(local_neighbor_radius))
        if local_neighbor_radius not in offsets_cache:
            offsets_cache[local_neighbor_radius] = np.arange(
                -local_neighbor_radius,
                local_neighbor_radius + 1,
                dtype=np.int64,
            )

        base_aud = float(_freq_to_aud(f0_now, scale=scale, fs=sr))
        base_idx = _nearest_sorted_idx(ref_fc_aud, base_aud)
        cand_idx = base_idx + offsets_cache[local_neighbor_radius]
        cand_idx = cand_idx[(cand_idx >= 1) & (cand_idx < (len(ref_fc_aud) - 1))]
        if cand_idx.size == 0:
            continue

        scores = []
        bins = []

        for idx in cand_idx:
            cand_aud = float(ref_fc_aud[int(idx)])
            cand_f0 = float(_aud_to_freq(cand_aud, scale=scale, fs=sr))
            if cand_f0 <= min_f0_hz or not np.isfinite(cand_f0):
                continue

            max_h = min(n_harmonics, int(nyquist // max(cand_f0, 1e-8)))
            if max_h < 1:
                continue

            score = 0.0
            for _, scale_power, _, scale_fc_aud, scale_frame_idx in scale_data:
                frame = int(scale_frame_idx[i])
                spec_frame = scale_power[:, frame]
                norm = float(np.mean(spec_frame)) + 1e-12

                for h in range(1, max_h + 1):
                    target_hz = float(h) * cand_f0
                    if target_hz >= nyquist:
                        break
                    target_aud = float(_freq_to_aud(target_hz, scale=scale, fs=sr))
                    h_idx = _nearest_sorted_idx(scale_fc_aud, target_aud)
                    lo = max(0, h_idx - harmonic_bin_radius)
                    hi = min(len(spec_frame) - 1, h_idx + harmonic_bin_radius)
                    band_val = float(np.mean(spec_frame[lo : hi + 1]))
                    h_weight = 1.0 / (float(h) ** harmonic_weight_exp)
                    score += h_weight * (band_val / norm)

            scores.append(score)
            bins.append(float(idx))

        if len(scores) == 0:
            continue

        scores = np.asarray(scores, dtype=np.float64)
        bins = np.asarray(bins, dtype=np.float64)
        best_i = int(np.argmax(scores))
        best_bin = float(bins[best_i])

        if subbin_interpolation and 0 < best_i < (len(scores) - 1):
            delta = _quadratic_peak_offset(
                y_left=float(scores[best_i - 1]),
                y_center=float(scores[best_i]),
                y_right=float(scores[best_i + 1]),
            )
            best_bin = float(np.clip(best_bin + delta, 1.0, float(len(ref_fc_aud) - 2)))

        best_aud = float(np.interp(best_bin, channel_pos, ref_fc_aud))
        f0_out[i] = float(_aud_to_freq(best_aud, scale=scale, fs=sr))

    if median_kernel > 1:
        f0_out = _smooth_voiced_median(f0_out, median_kernel)

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
    kernel_size: int,
    kernel_sizes: list[int] | None,
    num_channels: int,
    stride: int,
    elelet_fmin: float,
    elelet_fmax: float,
    supp_mult: float,
    scale: str,
    n_harmonics: int,
    neighbor_radius_bins: int,
    harmonic_bin_radius: int,
    power_exp: float,
    harmonic_weight_exp: float,
    min_f0_hz: float,
    median_kernel: int,
    subbin_interpolation: bool,
    use_precomputed_representations: str | Path | None = None,
) -> None:
    f0_dir = input_dir / f0_input_dir_name
    if not f0_dir.exists():
        raise FileNotFoundError(f"F0 input directory not found: {f0_dir}")

    csv_files = sorted(f0_dir.glob("*.f0.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No .f0.csv files found in {f0_dir}")

    output_dir = input_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    precomputed_dir = (
        resolve_precomputed_dir(use_precomputed_representations, "elelet")
        if use_precomputed_representations is not None
        else None
    )
    display_kernel_sizes = kernel_sizes if kernel_sizes else [kernel_size]
    display_stride = stride
    display_num_channels = num_channels
    if precomputed_dir is not None:
        params = read_metainfo(precomputed_dir).get("parameters", {})
        display_kernel_sizes = [int(params.get("kernel_size", kernel_size))]
        display_stride = int(params.get("stride", stride))
        display_num_channels = int(params.get("num_channels", num_channels))

    print(f"Input dir:   {input_dir}")
    print(f"F0 input:    {f0_dir}")
    print(f"F0 output:   {output_dir}")
    print(f"Files:       {len(csv_files)}")
    source = f"{precomputed_dir} (single scale)" if precomputed_dir is not None else "computed on the fly"
    print(f"Source:      {source}")
    print(
        "Elelet: "
        f"kernel_sizes={display_kernel_sizes}, "
        f"channels={display_num_channels}, stride={display_stride}, "
        f"f_min={elelet_fmin}, f_max={elelet_fmax}, supp_mult={supp_mult}, scale={scale}"
    )
    edge_frames = int(max(display_kernel_sizes) // 2 // max(display_stride, 1))
    print(
        "Params: "
        f"harmonics={n_harmonics}, neighbor_radius_bins={neighbor_radius_bins}, "
        f"harmonic_bin_radius={harmonic_bin_radius}, power_exp={power_exp}, "
        f"harmonic_weight_exp={harmonic_weight_exp}, min_f0_hz={min_f0_hz}, "
        f"median_kernel={median_kernel}, subbin_interpolation={subbin_interpolation}, "
        f"edge_frames_nb1={edge_frames}"
    )

    processed = 0
    skipped = 0
    delta_values = []

    for csv_path in tqdm(csv_files, desc="Refining f0 (harmonic Elelet)"):
        stem = csv_path.stem.replace(".f0", "")
        audio_path = input_dir / f"{stem}.wav"
        output_csv_path = output_dir / csv_path.name

        ok, mean_abs_delta = refine_f0_file(
            audio_path=audio_path,
            f0_csv_path=csv_path,
            output_csv_path=output_csv_path,
            sr=sr,
            kernel_size=kernel_size,
            kernel_sizes=kernel_sizes,
            num_channels=num_channels,
            stride=stride,
            elelet_fmin=elelet_fmin,
            elelet_fmax=elelet_fmax,
            supp_mult=supp_mult,
            scale=scale,
            n_harmonics=n_harmonics,
            neighbor_radius_bins=neighbor_radius_bins,
            harmonic_bin_radius=harmonic_bin_radius,
            power_exp=power_exp,
            harmonic_weight_exp=harmonic_weight_exp,
            min_f0_hz=min_f0_hz,
            median_kernel=median_kernel,
            subbin_interpolation=subbin_interpolation,
            precomputed_dir=precomputed_dir,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refine first-harmonic contours with Elelet harmonic salience "
            "using auditory-scale candidate search."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Directory with wav files.")
    parser.add_argument(
        "--f0_input_dir_name",
        type=str,
        default="f0_corrected",
        help="Input F0 subdirectory under input dir.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="f0_refined",
        help="Output subdirectory under input dir.",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument(
        "--use_precomputed_representations",
        type=str,
        default=None,
        help="Path to a precomputed Elelet folder, or a parent containing elelet_*.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=8000,
        help="Legacy single Elelet kernel size (used when --kernel_sizes is empty).",
    )
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="8000",
        help="Comma-separated Elelet kernel sizes for MSS-style scoring (e.g., 8000,16000,24000).",
    )
    parser.add_argument("--num_channels", type=int, default=2048, help="Elelet channels.")
    parser.add_argument("--stride", type=int, default=256, help="Elelet hop/stride in samples.")
    parser.add_argument("--elelet_fmin", type=float, default=5.0, help="Elelet min frequency (Hz).")
    parser.add_argument("--elelet_fmax", type=float, default=200.0, help="Elelet max frequency (Hz).")
    parser.add_argument("--supp_mult", type=float, default=0.2, help="Elelet support multiplier.")
    parser.add_argument(
        "--scale",
        type=str,
        default="elelog",
        help="Auditory scale for freqtoaud/audtofreq (e.g., elelog, mel, erb).",
    )
    parser.add_argument("--n_harmonics", type=int, default=9, help="Harmonics in salience score.")
    parser.add_argument(
        "--neighbor_radius_bins",
        type=int,
        default=5,
        help="Candidate search radius in auditory bins.",
    )
    parser.add_argument(
        "--harmonic_bin_radius",
        type=int,
        default=2,
        help="Per-harmonic channel averaging radius.",
    )
    parser.add_argument("--power_exp", type=float, default=2.0, help="Magnitude exponent.")
    parser.add_argument(
        "--harmonic_weight_exp",
        type=float,
        default=0.5,
        help="Harmonic weight exponent: weight=1/h^exp.",
    )
    parser.add_argument("--min_f0_hz", type=float, default=5.0, help="Minimum voiced F0 (Hz).")
    parser.add_argument(
        "--median_kernel",
        type=int,
        default=3,
        help="Median kernel on voiced regions (1 disables).",
    )
    parser.add_argument(
        "--subbin_interpolation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable quadratic sub-bin interpolation on candidate scores (default: enabled).",
    )

    args = parser.parse_args()
    kernel_sizes = _parse_int_list(args.kernel_sizes) if args.kernel_sizes.strip() else [args.kernel_size]
    refine_f0_folder(
        input_dir=Path(args.input),
        f0_input_dir_name=args.f0_input_dir_name,
        output_dir_name=args.output_dir_name,
        sr=args.sr,
        kernel_size=args.kernel_size,
        kernel_sizes=kernel_sizes,
        num_channels=args.num_channels,
        stride=args.stride,
        elelet_fmin=args.elelet_fmin,
        elelet_fmax=args.elelet_fmax,
        supp_mult=args.supp_mult,
        scale=args.scale,
        n_harmonics=args.n_harmonics,
        neighbor_radius_bins=args.neighbor_radius_bins,
        harmonic_bin_radius=args.harmonic_bin_radius,
        power_exp=args.power_exp,
        harmonic_weight_exp=args.harmonic_weight_exp,
        min_f0_hz=args.min_f0_hz,
        median_kernel=args.median_kernel,
        subbin_interpolation=args.subbin_interpolation,
        use_precomputed_representations=args.use_precomputed_representations,
    )


if __name__ == "__main__":
    main()
