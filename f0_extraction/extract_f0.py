"""
Extract first-harmonic (F1/H1) contours from 16 kHz audio files.

The main workflow is spectral peak tracking in one of two representations:

  stft   -> STFT magnitude peak tracking
  elelet -> Elelet magnitude peak tracking

The output CSV column is named `frequency` for compatibility with the rest of
the project, but it stores the tracked first harmonic by default. The refinement
step then converts H1 -> F0 internally and maps back to H1 for output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f0_extraction.pipeline import extracted_dir_name, normalize_algorithm


REPRESENTATIONS = ("elelet", "stft")


def normalize_representation(representation: str) -> str:
    value = representation.lower().strip()
    if value not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{representation}'. Expected one of: {', '.join(REPRESENTATIONS)}")
    return value


def _as_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    return audio


def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 10.0, order: int = 5) -> np.ndarray:
    """Remove DC offset and subsonic noise before low-frequency tracking."""
    if cutoff <= 0:
        return audio
    nyquist = sr / 2
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, audio)


def median_filter_pitch(
    frequency: np.ndarray,
    confidence: np.ndarray,
    kernel_size: int = 5,
    conf_threshold: float = 0.2,
) -> np.ndarray:
    """Median-filter confident voiced samples while leaving weak frames alone."""
    if kernel_size <= 1:
        return frequency
    if kernel_size % 2 == 0:
        kernel_size += 1

    mask = confidence > conf_threshold
    freq_filtered = frequency.copy()
    if mask.sum() > kernel_size:
        freq_filtered[mask] = median_filter(frequency[mask], size=kernel_size)
    return freq_filtered


def _nearest_allowed_peak(
    frame_coeffs: np.ndarray,
    freqs: np.ndarray,
    center_freq: float | None,
    max_jump: float | None,
) -> int:
    if center_freq is not None and np.isfinite(center_freq) and max_jump is not None:
        allowed = np.abs(freqs - center_freq) <= max_jump
        if np.any(allowed):
            masked = frame_coeffs.copy()
            masked[~allowed] = -np.inf
            return int(np.argmax(masked))
    return int(np.argmax(frame_coeffs))


def _track_spectral_peak(
    coeffs_abs: np.ndarray,
    freqs_hz: np.ndarray,
    times: np.ndarray,
    f1_min: float,
    f1_max: float,
    *,
    max_jump: float | None,
    use_global_peak: bool,
    energy_threshold: float,
    divide_by_2: bool,
    median_kernel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the dominant spectral peak in the F1 search band."""
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    coeffs_abs = np.asarray(coeffs_abs)
    freq_mask = (freqs_hz >= f1_min) & (freqs_hz <= f1_max)
    if not np.any(freq_mask):
        raise ValueError(f"No frequency bins found in F1 range {f1_min}-{f1_max} Hz")

    search_coeffs = coeffs_abs[freq_mask, :]
    search_freqs = freqs_hz[freq_mask]
    num_frames = search_coeffs.shape[1]
    tracked = np.zeros(num_frames, dtype=np.float64)
    confidence = np.zeros(num_frames, dtype=np.float64)
    peak_magnitudes = np.zeros(num_frames, dtype=np.float64)

    def set_frame(frame_idx: int, peak_idx: int) -> None:
        frame = search_coeffs[:, frame_idx]
        peak_freq = float(search_freqs[peak_idx])
        peak_mag = float(frame[peak_idx])
        tracked[frame_idx] = peak_freq
        peak_magnitudes[frame_idx] = peak_mag
        mean_energy = float(np.mean(frame))
        confidence[frame_idx] = peak_mag / (mean_energy * len(frame)) if mean_energy > 0 else 0.0

    if use_global_peak:
        global_peak_idx = int(np.argmax(np.mean(search_coeffs, axis=1)))
        global_peak_freq = float(search_freqs[global_peak_idx])
        start_frame = int(np.argmax(np.sum(search_coeffs, axis=0)))
        start_peak_idx = _nearest_allowed_peak(
            search_coeffs[:, start_frame],
            search_freqs,
            global_peak_freq,
            max_jump,
        )
        set_frame(start_frame, start_peak_idx)

        for frame_idx in range(start_frame + 1, num_frames):
            peak_idx = _nearest_allowed_peak(
                search_coeffs[:, frame_idx],
                search_freqs,
                tracked[frame_idx - 1],
                max_jump,
            )
            set_frame(frame_idx, peak_idx)

        for frame_idx in range(start_frame - 1, -1, -1):
            peak_idx = _nearest_allowed_peak(
                search_coeffs[:, frame_idx],
                search_freqs,
                tracked[frame_idx + 1],
                max_jump,
            )
            set_frame(frame_idx, peak_idx)
    else:
        for frame_idx in range(num_frames):
            center_freq = tracked[frame_idx - 1] if frame_idx > 0 and tracked[frame_idx - 1] > 0 else None
            peak_idx = _nearest_allowed_peak(
                search_coeffs[:, frame_idx],
                search_freqs,
                center_freq,
                max_jump,
            )
            set_frame(frame_idx, peak_idx)

    if median_kernel > 1:
        smoothed = median_filter(tracked, size=median_kernel)
        tracked[tracked > 0] = smoothed[tracked > 0]

    if np.any(peak_magnitudes > 0):
        threshold = energy_threshold * float(np.max(peak_magnitudes))
        low_energy = peak_magnitudes < threshold
        tracked[low_energy] = 0.0
        confidence[low_energy] = 0.0

    if np.any(confidence > 0):
        conf_threshold = float(np.percentile(confidence[confidence > 0], 25))
        tracked = median_filter_pitch(tracked, confidence, kernel_size=median_kernel, conf_threshold=conf_threshold)

    if divide_by_2:
        tracked = tracked / 2.0

    return times[:num_frames], tracked, confidence


def extract_f1_stft(
    audio: np.ndarray,
    sr: int = 16000,
    frame_resolution: float = 0.016,
    f1_min: float = 22.5,
    f1_max: float = 50.0,
    n_fft: int = 8192,
    win_length: int | None = None,
    max_jump: float | None = 1.0,
    use_global_peak: bool = True,
    energy_threshold: float = 0.2,
    divide_by_2: bool = False,
    median_kernel: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the F1 spectral peak in an STFT magnitude spectrogram."""
    audio = highpass_filter(_as_mono(audio), sr, cutoff=f1_min * 0.5)
    hop_length = int(round(sr * frame_resolution))
    stft = librosa.stft(
        y=audio,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = np.arange(magnitude.shape[1], dtype=np.float64) * hop_length / sr
    return _track_spectral_peak(
        magnitude,
        freqs,
        times,
        f1_min,
        f1_max,
        max_jump=max_jump,
        use_global_peak=use_global_peak,
        energy_threshold=energy_threshold,
        divide_by_2=divide_by_2,
        median_kernel=median_kernel,
    )


def extract_f1_elelet(
    audio: np.ndarray,
    sr: int = 16000,
    stride: int = 256,
    f1_min: float = 22.5,
    f1_max: float = 50.0,
    num_channels: int = 1024,
    kernel_size: int = 24000,
    transform_fmin: float = 5.0,
    transform_fmax: float = 100.0,
    supp_mult: float = 0.2,
    scale: str = "elelog",
    max_jump: float | None = 1.0,
    use_global_peak: bool = True,
    energy_threshold: float = 0.2,
    divide_by_2: bool = False,
    median_kernel: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the F1 spectral peak in an Elelet magnitude representation."""
    from tf_transforms.transforms import Elelet

    audio = _as_mono(audio)
    transform = Elelet(
        kernel_size=kernel_size,
        num_channels=num_channels,
        stride=stride,
        fmin=transform_fmin,
        fmax=max(transform_fmax, f1_max),
        fs=sr,
        supp_mult=supp_mult,
        scale=scale,
    )
    coeffs_abs = np.abs(transform(audio))
    freqs = transform.fc
    if hasattr(freqs, "numpy"):
        freqs = freqs.numpy()
    freqs = np.asarray(freqs, dtype=np.float64)[: coeffs_abs.shape[0]]
    times = np.arange(coeffs_abs.shape[1], dtype=np.float64) * stride / sr
    return _track_spectral_peak(
        coeffs_abs,
        freqs,
        times,
        f1_min,
        f1_max,
        max_jump=max_jump,
        use_global_peak=use_global_peak,
        energy_threshold=energy_threshold,
        divide_by_2=divide_by_2,
        median_kernel=median_kernel,
    )


def extract_f0_pyin(
    audio: np.ndarray,
    sr: int = 16000,
    frame_resolution: float = 0.016,
    fmin: float = 5.0,
    fmax: float = 50.0,
    extract_f1: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy pYIN extractor kept for comparisons with older experiments."""
    audio_filtered = highpass_filter(_as_mono(audio), sr, cutoff=fmin * 0.5)
    hop_length = int(round(sr * frame_resolution))

    if extract_f1:
        pyin_fmin = fmin * 2.0
        pyin_fmax = fmax * 2.0
        frame_length = 4096
        frequency, _, confidence = librosa.pyin(
            audio_filtered,
            fmin=pyin_fmin,
            fmax=pyin_fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
        )
    else:
        frame_length = 8192
        frequency, _, confidence = librosa.pyin(
            audio_filtered,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
        )

    frequency = np.nan_to_num(frequency, nan=0.0)
    confidence = np.nan_to_num(confidence, nan=0.0)
    frequency = median_filter_pitch(frequency, confidence, kernel_size=5, conf_threshold=0.1)
    time = np.arange(len(frequency), dtype=np.float64) * hop_length / sr
    return time, frequency, confidence


# Backwards-compatible names used by notebooks or older scripts.
extract_f0_elelet = extract_f1_elelet


def extract_f0_from_dataset(
    audio_dir: str | Path,
    sr: int = 16000,
    frame_resolution: float = 0.016,
    pipeline: str = "elelet",
    algorithm_name: str | None = None,
    output_dir_name: str | None = None,
    method: str = "spectral",
    f1_min: float = 22.5,
    f1_max: float = 50.0,
    max_jump: float | None = 1.0,
    use_global_peak: bool = True,
    energy_threshold: float = 0.2,
    divide_by_2: bool = False,
    median_kernel: int = 5,
    stft_n_fft: int = 8192,
    stft_win_length: int | None = None,
    elelet_num_channels: int = 1024,
    elelet_kernel_size: int = 24000,
    elelet_fmin: float = 5.0,
    elelet_fmax: float = 100.0,
    elelet_supp_mult: float = 0.2,
    elelet_scale: str = "elelog",
) -> None:
    """Extract F1 contours for all WAV files in a directory."""
    pipeline = normalize_representation(pipeline)
    algorithm_name = normalize_algorithm(algorithm_name or ("pyin" if method == "pyin" else pipeline))
    audio_dir = Path(audio_dir)
    output_dir = audio_dir / (output_dir_name or extracted_dir_name(algorithm_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    if len(audio_files) == 0:
        print(f"ERROR: No WAV files found in {audio_dir}")
        return

    print("=" * 60)
    print("F1 SPECTRAL PEAK EXTRACTION")
    print("=" * 60)
    print(f"Representation: {pipeline}")
    print(f"Algorithm: {algorithm_name}")
    print(f"Input:    {audio_dir}")
    print(f"Output:   {output_dir}")
    print(f"Files:    {len(audio_files)}")
    print(f"F1 band:  {f1_min:g}-{f1_max:g} Hz")
    print(f"Timing:   sr={sr}, frame_resolution={frame_resolution:g}s")
    print(f"Tracking: global_peak={use_global_peak}, max_jump={max_jump}, energy_threshold={energy_threshold:g}")
    if divide_by_2:
        print("Output:   frequency column stores F1/2")
    else:
        print("Output:   frequency column stores F1/H1")
    print("=" * 60 + "\n")

    successful = 0
    failed: list[str] = []

    for audio_path in tqdm(audio_files, desc=f"Extracting F1 ({pipeline})"):
        try:
            audio, file_sr = sf.read(str(audio_path))
            audio = _as_mono(audio)
            if file_sr != sr:
                print(f"\n  WARNING: {audio_path.name} has SR={file_sr}, expected {sr}; resampling")
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

            if method == "pyin":
                time, frequency, confidence = extract_f0_pyin(
                    audio,
                    sr=sr,
                    frame_resolution=frame_resolution,
                    fmin=f1_min * 0.5,
                    fmax=f1_max * 0.5,
                    extract_f1=True,
                )
            elif pipeline == "stft":
                time, frequency, confidence = extract_f1_stft(
                    audio,
                    sr=sr,
                    frame_resolution=frame_resolution,
                    f1_min=f1_min,
                    f1_max=f1_max,
                    n_fft=stft_n_fft,
                    win_length=stft_win_length,
                    max_jump=max_jump,
                    use_global_peak=use_global_peak,
                    energy_threshold=energy_threshold,
                    divide_by_2=divide_by_2,
                    median_kernel=median_kernel,
                )
            else:
                stride = int(round(sr * frame_resolution))
                time, frequency, confidence = extract_f1_elelet(
                    audio,
                    sr=sr,
                    stride=stride,
                    f1_min=f1_min,
                    f1_max=f1_max,
                    num_channels=elelet_num_channels,
                    kernel_size=elelet_kernel_size,
                    transform_fmin=elelet_fmin,
                    transform_fmax=elelet_fmax,
                    supp_mult=elelet_supp_mult,
                    scale=elelet_scale,
                    max_jump=max_jump,
                    use_global_peak=use_global_peak,
                    energy_threshold=energy_threshold,
                    divide_by_2=divide_by_2,
                    median_kernel=median_kernel,
                )

            nonzero = np.where(frequency > 0)[0]
            start_time = float(time[nonzero[0]]) if len(nonzero) else 0.0
            end_time = float(time[nonzero[-1]]) if len(nonzero) else 0.0

            df = pd.DataFrame(
                {
                    "time": time,
                    "frequency": frequency,
                    "confidence": confidence,
                    "start_point": np.full(len(time), start_time),
                    "end_point": np.full(len(time), end_time),
                    "representation": pipeline,
                    "algorithm": algorithm_name,
                    "frequency_role": "f1_div_2" if divide_by_2 else "f1",
                }
            )
            df.to_csv(output_dir / f"{audio_path.stem}.f0.csv", index=False, float_format="%.6f")
            successful += 1
        except Exception as exc:
            print(f"\nError processing {audio_path.name}: {exc}")
            failed.append(audio_path.name)

    print("\n" + "=" * 60)
    print("F1 EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {successful}/{len(audio_files)} files")
    if failed:
        print(f"Failed: {len(failed)} files")
        for name in failed:
            print(f"  - {name}")
    print(f"Output: {output_dir}")


def _legacy_pipeline_from_args(args: argparse.Namespace) -> str:
    if args.use_elelet:
        return "elelet"
    return args.pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract F1 contours by spectral peak tracking in STFT or Elelet space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended workflows:
  python f0_extraction/extract_f0.py --input data/rumbles --pipeline elelet
  python f0_extraction/extract_f0.py --input data/rumbles --pipeline stft

Outputs:
  elelet -> data/rumbles/f0_elelet/*.f0.csv
  stft   -> data/rumbles/f0_stft/*.f0.csv
        """,
    )
    parser.add_argument("--input", type=str, required=True, help="Directory with WAV files.")
    parser.add_argument("--pipeline", choices=("elelet", "stft"), default="elelet", help="Representation to track in.")
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default=None,
        help="Label used in output folders for algorithm comparisons (default: peak).",
    )
    parser.add_argument("--output_dir_name", type=str, default=None, help="Output directory under input dir.")
    parser.add_argument("--method", choices=("spectral", "pyin"), default="spectral", help="Extraction method.")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--frame_resolution", type=float, default=0.016, help="Frame resolution in seconds.")
    parser.add_argument("--f1_min", type=float, default=22.5, help="Minimum F1 frequency to track.")
    parser.add_argument("--f1_max", type=float, default=50.0, help="Maximum F1 frequency to track.")
    parser.add_argument("--max_jump", type=float, default=1.0, help="Maximum frame-to-frame F1 jump in Hz.")
    parser.add_argument("--no_max_jump", dest="max_jump", action="store_const", const=None, help="Disable continuity limit.")
    parser.add_argument("--global_peak", action=argparse.BooleanOptionalAction, default=True, help="Track outward from strongest frame.")
    parser.add_argument("--energy_threshold", type=float, default=0.2, help="Fraction of max peak magnitude used to zero weak frames.")
    parser.add_argument("--divide_by_2", action="store_true", help="Store F1/2 instead of F1.")
    parser.add_argument("--median_kernel", type=int, default=5, help="Median smoothing kernel.")

    parser.add_argument("--stft_n_fft", type=int, default=8192, help="STFT FFT size.")
    parser.add_argument("--stft_win_length", type=int, default=None, help="STFT window length.")

    parser.add_argument("--elelet_num_channels", type=int, default=1024, help="Elelet channels.")
    parser.add_argument("--elelet_kernel_size", type=int, default=24000, help="Elelet kernel size.")
    parser.add_argument("--elelet_transform_fmin", type=float, default=5.0, help="Elelet transform minimum frequency.")
    parser.add_argument("--elelet_transform_fmax", type=float, default=100.0, help="Elelet transform maximum frequency.")
    parser.add_argument("--elelet_supp_mult", type=float, default=0.2, help="Elelet support multiplier.")
    parser.add_argument("--elelet_scale", type=str, default="elelog", help="Elelet frequency scale.")

    # Legacy aliases from the previous script.
    parser.add_argument("--use_elelet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--elelet_fmin", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--elelet_fmax", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--elelet_max_jump", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--elelet_use_global_peak", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--elelet_energy_threshold", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.elelet_fmin is not None:
        args.f1_min = args.elelet_fmin
    if args.elelet_fmax is not None:
        args.f1_max = args.elelet_fmax
    if args.elelet_max_jump is not None:
        args.max_jump = args.elelet_max_jump
    if args.elelet_use_global_peak:
        args.global_peak = True
    if args.elelet_energy_threshold is not None:
        args.energy_threshold = args.elelet_energy_threshold

    extract_f0_from_dataset(
        audio_dir=args.input,
        sr=args.sr,
        frame_resolution=args.frame_resolution,
        pipeline=_legacy_pipeline_from_args(args),
        algorithm_name=args.algorithm_name,
        output_dir_name=args.output_dir_name,
        method=args.method,
        f1_min=args.f1_min,
        f1_max=args.f1_max,
        max_jump=args.max_jump,
        use_global_peak=args.global_peak,
        energy_threshold=args.energy_threshold,
        divide_by_2=args.divide_by_2,
        median_kernel=args.median_kernel,
        stft_n_fft=args.stft_n_fft,
        stft_win_length=args.stft_win_length,
        elelet_num_channels=args.elelet_num_channels,
        elelet_kernel_size=args.elelet_kernel_size,
        elelet_fmin=args.elelet_transform_fmin,
        elelet_fmax=args.elelet_transform_fmax,
        elelet_supp_mult=args.elelet_supp_mult,
        elelet_scale=args.elelet_scale,
    )


if __name__ == "__main__":
    main()
