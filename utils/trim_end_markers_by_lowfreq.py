"""
Adjust end markers in f0_corrected based on low-frequency STFT energy.

Workflow:
1) Trim audio to existing start/end markers in f0_corrected.
2) Compute sum of STFT magnitude from 0..max_hz for each frame.
3) If two voiced segments are detected, reset end_point to end of the first.
"""

import argparse
from pathlib import Path

import librosa
import librosa.display
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_lowfreq_energy(audio, sr, max_hz, hop_length, n_fft, smooth_frames):
    """Return low-frequency energy per STFT frame (smoothed)."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_idx = freqs <= max_hz
    energy = np.sum(np.abs(stft[low_idx, :]), axis=0)

    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        energy = np.convolve(energy, kernel, mode="same")

    return energy


def find_segments(mask):
    """Return list of (start_idx, end_idx) for contiguous True regions."""
    segments = []
    if mask.size == 0:
        return segments

    in_seg = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_seg:
            in_seg = True
            start = i
        elif not val and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask) - 1))
    return segments


def pick_new_end(
    segments,
    hop_length,
    sr,
    min_seg_frames,
    min_gap_frames,
    min_first_frames,
    min_second_frames,
    max_second_frames,
    require_exact_two,
):
    """Return end time (seconds) for the first segment if a short second exists."""
    if len(segments) < 2:
        return None
    if require_exact_two and len(segments) != 2:
        return None

    for i in range(len(segments) - 1):
        start_a, end_a = segments[i]
        start_b, end_b = segments[i + 1]
        frames_a = end_a - start_a + 1
        frames_b = end_b - start_b + 1
        gap_frames = start_b - end_a - 1

        if frames_a < min_seg_frames or frames_b < min_seg_frames:
            continue
        if frames_a < min_first_frames:
            continue
        if frames_b < min_second_frames or frames_b > max_second_frames:
            continue
        if gap_frames < min_gap_frames:
            continue

        end_time = (end_a * hop_length) / sr
        return end_time
    return None


def adjust_markers(
    audio_dir,
    f0_dir=None,
    max_hz=100.0,
    hop_length_s=0.016,
    n_fft=4096,
    smooth_s=0.08,
    threshold_ratio=0.35,
    min_segment_s=0.12,
    min_gap_s=0.12,
    min_first_segment_s=0.25,
    min_second_segment_s=0.06,
    max_second_segment_s=0.25,
    require_exact_two=True,
    preview_dir=None,
    preview_limit=50,
    interactive=False,
    write=False,
):
    audio_dir = Path(audio_dir)
    f0_dir = Path(f0_dir) if f0_dir else audio_dir / "f0_corrected"

    if not f0_dir.exists():
        raise FileNotFoundError(f"f0_corrected directory not found: {f0_dir}")

    f0_files = sorted(f0_dir.glob("*.f0.csv"))
    if not f0_files:
        raise FileNotFoundError(f"No .f0.csv files found in {f0_dir}")

    print("=" * 60)
    print("ADJUST END MARKERS BY LOW-FREQ ENERGY")
    print("=" * 60)
    print(f"Audio directory: {audio_dir}")
    print(f"F0 directory:    {f0_dir}")
    print(f"Files to scan:   {len(f0_files)}")
    print(f"Max Hz:          {max_hz}")
    print(f"Hop length:      {hop_length_s:.3f}s")
    print(f"n_fft:           {n_fft}")
    print(f"Smoothing:       {smooth_s:.3f}s")
    print(f"Threshold ratio: {threshold_ratio}")
    print(f"Min segment:     {min_segment_s:.3f}s")
    print(f"Min gap:         {min_gap_s:.3f}s")
    print(f"Min first seg:   {min_first_segment_s:.3f}s")
    print(f"Min second seg:  {min_second_segment_s:.3f}s")
    print(f"Max second seg:  {max_second_segment_s:.3f}s")
    print(f"Exact two segs:  {require_exact_two}")
    print(f"Preview dir:     {preview_dir}")
    print(f"Preview limit:   {preview_limit}")
    print(f"Interactive:     {interactive}")
    print(f"Write changes:   {write}")
    print()

    updated = 0
    skipped = 0
    errors = 0
    previewed = 0
    accepted = 0
    rejected = 0

    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

    for f0_path in tqdm(f0_files, desc="Analyzing"):
        try:
            audio_path = audio_dir / f"{f0_path.stem.replace('.f0', '')}.wav"
            if not audio_path.exists():
                skipped += 1
                continue

            df = pd.read_csv(f0_path)
            if "start_point" not in df.columns or "end_point" not in df.columns:
                skipped += 1
                continue

            start_mask = df["start_point"] == 1
            end_mask = df["end_point"] == 1
            if not start_mask.any() or not end_mask.any():
                skipped += 1
                continue

            t_start = df.loc[start_mask, "time"].values[0]
            t_end = df.loc[end_mask, "time"].values[0]
            if t_start > t_end:
                t_start, t_end = t_end, t_start

            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            start_sample = max(0, int(t_start * sr))
            end_sample = min(len(audio), int(t_end * sr))
            if end_sample <= start_sample:
                skipped += 1
                continue

            trimmed = audio[start_sample:end_sample]
            hop_length = max(1, int(sr * hop_length_s))
            smooth_frames = max(1, int(smooth_s / hop_length_s))
            min_seg_frames = max(1, int(min_segment_s / hop_length_s))
            min_gap_frames = max(1, int(min_gap_s / hop_length_s))
            min_first_frames = max(1, int(min_first_segment_s / hop_length_s))
            min_second_frames = max(1, int(min_second_segment_s / hop_length_s))
            max_second_frames = max(1, int(max_second_segment_s / hop_length_s))

            energy = compute_lowfreq_energy(
                trimmed,
                sr,
                max_hz=max_hz,
                hop_length=hop_length,
                n_fft=n_fft,
                smooth_frames=smooth_frames,
            )

            if energy.size == 0 or np.max(energy) <= 0:
                skipped += 1
                continue

            threshold = np.max(energy) * threshold_ratio
            mask = energy >= threshold
            segments = find_segments(mask)

            new_end_rel = pick_new_end(
                segments,
                hop_length=hop_length,
                sr=sr,
                min_seg_frames=min_seg_frames,
                min_gap_frames=min_gap_frames,
                min_first_frames=min_first_frames,
                min_second_frames=min_second_frames,
                max_second_frames=max_second_frames,
                require_exact_two=require_exact_two,
            )

            if new_end_rel is None:
                continue

            new_end = t_start + new_end_rel
            if new_end <= t_start:
                continue

            eligible = df["time"] <= new_end
            if not eligible.any():
                continue

            end_idx = df.loc[eligible, "time"].idxmax()
            if df.loc[end_idx, "end_point"] == 1:
                continue

            df.loc[:, "end_point"] = 0
            df.loc[end_idx, "end_point"] = 1

            show_preview = interactive or (preview_dir and previewed < preview_limit)
            if show_preview:
                try:
                    hop_len = hop_length
                    stft = librosa.stft(trimmed, n_fft=n_fft, hop_length=hop_len)
                    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

                    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
                    librosa.display.specshow(
                        stft_db,
                        sr=sr,
                        hop_length=hop_len,
                        x_axis="time",
                        y_axis="hz",
                        ax=ax,
                        cmap="magma",
                    )
                    ax.set_ylim(0, max_hz * 6)

                    # Overlay f0 in trimmed window
                    f0_mask = (df["time"] >= t_start) & (df["time"] <= t_end)
                    f0_times = df.loc[f0_mask, "time"].to_numpy() - t_start
                    f0_vals = df.loc[f0_mask, "frequency"].to_numpy()
                    ax.plot(f0_times, f0_vals, color="cyan", linewidth=1.0, alpha=0.8)

                    # Mark original end and new end
                    ax.axvline(t_end - t_start, color="white", linestyle="--", linewidth=1.0)
                    ax.axvline(new_end - t_start, color="lime", linestyle="-", linewidth=1.2)
                    ax.set_xlim(0, t_end - t_start)

                    ax.set_title(
                        f"{f0_path.stem}  end {t_end:.3f}s -> {df.loc[end_idx, 'time']:.3f}s"
                    )

                    if preview_dir and previewed < preview_limit:
                        out_path = preview_dir / f"{f0_path.stem}.png"
                        fig.savefig(out_path, dpi=150)
                        previewed += 1

                    if interactive:
                        plt.show(block=True)
                        resp = input("Accept change? [y/N] ").strip().lower()
                        if resp != "y":
                            rejected += 1
                            plt.close(fig)
                            continue
                        accepted += 1
                    plt.close(fig)
                except Exception as exc:
                    print(f"\n  Preview error for {f0_path.name}: {exc}")
                    if interactive:
                        rejected += 1
                    continue

            if write:
                df.to_csv(f0_path, index=False)

            updated += 1
            print(
                f"{f0_path.stem}: end {t_end:.3f}s -> {df.loc[end_idx, 'time']:.3f}s"
            )

        except Exception as exc:
            print(f"\n  Error processing {f0_path.name}: {exc}")
            errors += 1
            continue

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")
    if preview_dir:
        print(f"Previews: {previewed}")
    if interactive:
        print(f"Accepted: {accepted}")
        print(f"Rejected: {rejected}")
    if not write:
        print("\nDry run only. Use --write to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Adjust end markers in f0_corrected based on low-frequency STFT energy."
    )
    parser.add_argument("--input", required=True, help="Audio directory (wav + f0_corrected/)")
    parser.add_argument("--f0-dir", default=None, help="Override f0_corrected directory")
    parser.add_argument("--max-hz", type=float, default=100.0, help="Max frequency to sum")
    parser.add_argument("--hop-length-s", type=float, default=0.016, help="Hop length (seconds)")
    parser.add_argument("--n-fft", type=int, default=4096, help="STFT FFT size")
    parser.add_argument("--smooth-s", type=float, default=0.08, help="Smoothing window (seconds)")
    parser.add_argument("--threshold-ratio", type=float, default=0.35, help="Energy threshold ratio")
    parser.add_argument("--min-segment-s", type=float, default=0.12, help="Min segment duration (s)")
    parser.add_argument("--min-gap-s", type=float, default=0.12, help="Min gap between segments (s)")
    parser.add_argument("--min-first-segment-s", type=float, default=0.25, help="Min first segment (s)")
    parser.add_argument("--min-second-segment-s", type=float, default=0.06, help="Min second segment (s)")
    parser.add_argument("--max-second-segment-s", type=float, default=0.25, help="Max second segment (s)")
    parser.add_argument("--allow-multi-segments", action="store_true", help="Allow >2 segments")
    parser.add_argument("--preview-dir", type=str, default=None, help="Directory to save preview PNGs")
    parser.add_argument("--preview-limit", type=int, default=50, help="Max previews to save")
    parser.add_argument("--interactive", action="store_true", help="Show spectrograms and ask y/n")
    parser.add_argument("--write", action="store_true", help="Write updated CSVs")

    args = parser.parse_args()
    adjust_markers(
        audio_dir=args.input,
        f0_dir=args.f0_dir,
        max_hz=args.max_hz,
        hop_length_s=args.hop_length_s,
        n_fft=args.n_fft,
        smooth_s=args.smooth_s,
        threshold_ratio=args.threshold_ratio,
        min_segment_s=args.min_segment_s,
        min_gap_s=args.min_gap_s,
        min_first_segment_s=args.min_first_segment_s,
        min_second_segment_s=args.min_second_segment_s,
        max_second_segment_s=args.max_second_segment_s,
        require_exact_two=not args.allow_multi_segments,
        preview_dir=args.preview_dir,
        preview_limit=args.preview_limit,
        interactive=args.interactive,
        write=args.write,
    )


if __name__ == "__main__":
    main()
