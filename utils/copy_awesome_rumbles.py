#!/usr/bin/env python3
"""
Copy rumbles by category into a separate folder.

Category is derived from f0_corrected CSV columns:
- awesome: all rows have awesome=1
- overlapping: all rows have overlapping=1
- bad: all rows have bad=1
- average: none of the above
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


def get_file_category(csv_path):
    df = pd.read_csv(csv_path)

    is_awesome = ("awesome" in df.columns) and (df["awesome"].min() == 1)
    is_overlapping = ("overlapping" in df.columns) and (df["overlapping"].min() == 1)
    is_bad = ("bad" in df.columns) and (df["bad"].min() == 1)

    if is_awesome:
        return "awesome"
    if is_overlapping:
        return "overlapping"
    if is_bad:
        return "bad"
    return "average"


def get_marker_times(csv_path):
    df = pd.read_csv(csv_path)
    if "start_point" not in df.columns or "end_point" not in df.columns:
        return None, None

    start_rows = df[df["start_point"] == 1]
    end_rows = df[df["end_point"] == 1]
    if start_rows.empty or end_rows.empty:
        return None, None

    t_start = float(start_rows.iloc[0]["time"])
    t_end = float(end_rows.iloc[0]["time"])
    if t_start > t_end:
        t_start, t_end = t_end, t_start
    return t_start, t_end


def trim_f0_dataframe(csv_path, t_start, t_end):
    df = pd.read_csv(csv_path)
    time_mask = (df["time"] >= t_start) & (df["time"] <= t_end)
    df_trimmed = df[time_mask].copy()
    if "time" in df_trimmed.columns:
        df_trimmed["time"] = df_trimmed["time"] - t_start
    return df_trimmed


def find_audio_path(audio_dir, csv_path):
    base_name = csv_path.stem.replace(".f0", "")
    candidates = [
        audio_dir / f"{base_name}.wav",
        audio_dir / f"{base_name}.WAV",
        audio_dir / f"{base_name}.flac",
        audio_dir / f"{base_name}.mp3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(audio_dir.glob(f"{base_name}.*"))
    return matches[0] if matches else None


def get_duration_seconds(audio_path):
    try:
        info = sf.info(str(audio_path))
        if info.frames > 0 and info.samplerate > 0:
            return info.frames / info.samplerate
    except Exception:
        return None
    return None


def trim_audio_to_markers(audio_path, t_start, t_end):
    audio, sr = sf.read(str(audio_path))
    start_sample = int(max(0, t_start) * sr)
    end_sample = int(max(0, t_end) * sr)
    end_sample = min(len(audio), max(start_sample, end_sample))
    trimmed_audio = audio[start_sample:end_sample]
    return trimmed_audio, sr


def apply_bandpass_filter(audio, sample_rate, low_cutoff=10, high_cutoff=500, order=4):
    nyquist = sample_rate / 2
    low_cutoff = max(1, min(low_cutoff, nyquist - 1))
    high_cutoff = max(low_cutoff + 1, min(high_cutoff, nyquist - 1))

    sos = signal.butter(
        order,
        [low_cutoff / nyquist, high_cutoff / nyquist],
        btype="band",
        output="sos",
    )
    filtered_audio = signal.sosfilt(sos, audio)
    return filtered_audio.astype(np.float32)


def normalize_audio(audio, mode, target_rms=0.1):
    if mode == "none":
        return audio

    audio = audio.astype(np.float32, copy=False)
    if mode == "peak":
        peak = np.max(np.abs(audio)) if audio.size else 0.0
        if peak <= 0:
            return audio
        return audio / peak
    if mode == "rms":
        rms = np.sqrt(np.mean(audio ** 2)) if audio.size else 0.0
        if rms <= 0:
            return audio
        return audio * (target_rms / rms)

    raise ValueError(f"Unsupported normalization mode: {mode}")


def parse_categories(value):
    if value.strip().lower() == "all":
        return {"awesome", "overlapping", "bad", "average"}
    categories = {v.strip().lower() for v in value.split(",") if v.strip()}
    valid = {"awesome", "overlapping", "bad", "average"}
    invalid = categories - valid
    if invalid:
        raise ValueError(f"Invalid categories: {sorted(invalid)}")
    return categories


def main():
    parser = argparse.ArgumentParser(
        description="Copy rumbles by category into a separate folder."
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/rumbles",
        help="Directory containing audio files and f0_corrected/",
    )
    parser.add_argument(
        "--f0-dir",
        type=str,
        default=None,
        help="Directory containing f0_corrected CSV files (default: audio-dir/f0_corrected)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rumbles_awesome",
        help="Destination directory for copied audio files",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="awesome",
        help="Category to include: awesome, overlapping, bad, average, or all "
        "(comma-separated also supported)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional max duration in seconds to keep",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Optional min duration in seconds to keep",
    )
    parser.add_argument(
        "--include-f0",
        action="store_true",
        help="Also copy matching f0_corrected CSV files into output-dir/f0_corrected",
    )
    parser.add_argument(
        "--trim-to-markers",
        action="store_true",
        help="Trim audio (and f0 CSV if included) to start/end markers",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        help="Audio normalization: none, peak, or rms",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.01,
        help="Target RMS for rms normalization",
    )
    parser.add_argument(
        "--bandpass",
        action="store_true",
        help="Apply the same Butterworth bandpass as training",
    )
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=5.0,
        help="Bandpass low cutoff in Hz",
    )
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=2000.0,
        help="Bandpass high cutoff in Hz",
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    f0_dir = Path(args.f0_dir) if args.f0_dir else audio_dir / "f0_corrected"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not f0_dir.exists():
        raise FileNotFoundError(f"f0_corrected directory not found: {f0_dir}")

    categories = parse_categories(args.category)

    if args.include_f0:
        (output_dir / "f0_corrected").mkdir(parents=True, exist_ok=True)

    f0_files = sorted(f0_dir.glob("*.csv"))
    if len(f0_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {f0_dir}")

    copied = 0
    skipped = 0
    missing_audio = 0
    missing_markers = 0

    normalize_mode = args.normalize.strip().lower()
    if normalize_mode not in {"none", "peak", "rms"}:
        raise ValueError("normalize must be one of: none, peak, rms")

    for csv_file in f0_files:
        if get_file_category(csv_file) not in categories:
            continue

        audio_path = find_audio_path(audio_dir, csv_file)
        if audio_path is None:
            missing_audio += 1
            continue

        t_start, t_end = (None, None)
        if args.trim_to_markers:
            t_start, t_end = get_marker_times(csv_file)
            if t_start is None or t_end is None:
                missing_markers += 1
                continue

        if args.trim_to_markers:
            duration = max(0.0, t_end - t_start)
        else:
            duration = get_duration_seconds(audio_path)
        if duration is None:
            skipped += 1
            continue
        if args.min_duration is not None and duration < args.min_duration:
            skipped += 1
            continue
        if args.max_duration is not None and duration > args.max_duration:
            skipped += 1
            continue

        needs_processing = args.trim_to_markers or args.bandpass or normalize_mode != "none"
        if needs_processing:
            if args.trim_to_markers:
                audio, sr = trim_audio_to_markers(audio_path, t_start, t_end)
            else:
                audio, sr = sf.read(str(audio_path))

            if len(audio) == 0:
                skipped += 1
                continue

            if args.bandpass:
                audio = apply_bandpass_filter(
                    audio,
                    sr,
                    low_cutoff=args.bandpass_low,
                    high_cutoff=args.bandpass_high,
                )

            if normalize_mode != "none":
                audio = normalize_audio(audio, normalize_mode, args.target_rms)

            sf.write(str(output_dir / audio_path.name), audio, sr)
        else:
            shutil.copy2(audio_path, output_dir / audio_path.name)

        if args.include_f0:
            if args.trim_to_markers:
                df_trimmed = trim_f0_dataframe(csv_file, t_start, t_end)
                df_trimmed.to_csv(output_dir / "f0_corrected" / csv_file.name, index=False)
            else:
                shutil.copy2(csv_file, output_dir / "f0_corrected" / csv_file.name)
        copied += 1

    print(f"Copied: {copied}")
    if skipped:
        print(f"Skipped (duration filter or unreadable): {skipped}")
    if missing_markers:
        print(f"Skipped (missing markers): {missing_markers}")
    if missing_audio:
        print(f"Missing audio files: {missing_audio}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
