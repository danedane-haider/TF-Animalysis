"""
Compute mean harmonic distribution from the dataset to use as a data-driven prior.

This replaces the hand-crafted prior (F0=0.6, F1=1.0, etc.) with statistics
computed from the actual elephant rumble recordings.
"""

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def normalize_by_f1_peak(audio, f1_values, sample_rate, bandwidth_ratio=0.3):
    """
    Standalone implementation of F1-based peak normalization.
    (Copied logic to avoid importing from dataset module)

    Normalize audio by the peak amplitude in the F1 frequency band.

    Args:
        audio: Audio signal (numpy array)
        f1_values: F1 frequency values corresponding to frames (numpy array)
        sample_rate: Sample rate in Hz
        bandwidth_ratio: Bandwidth around F1 as a ratio (e.g., 0.3 = ±30% of F1)

    Returns:
        normalized_audio: Normalized audio
        f1_peak: The F1 peak value used for normalization
    """
    # Get median F1 for this segment (ignore zeros)
    f1_nonzero = f1_values[f1_values > 0]
    if len(f1_nonzero) == 0:
        # No voiced segments, fall back to standard peak normalization
        peak = np.abs(audio).max()
        if peak > 1e-8:
            return audio / peak, peak
        return audio, 1.0

    median_f1 = np.median(f1_nonzero)

    # Define bandpass filter around F1
    low_cutoff = median_f1 * (1 - bandwidth_ratio)
    high_cutoff = median_f1 * (1 + bandwidth_ratio)

    # Ensure cutoffs are valid
    low_cutoff = max(10, low_cutoff)  # At least 10 Hz
    high_cutoff = min(sample_rate / 2 - 10, high_cutoff)  # Below Nyquist

    # Design bandpass filter (4th order Butterworth)
    nyquist = sample_rate / 2
    sos = signal.butter(4, [low_cutoff / nyquist, high_cutoff / nyquist],
                       btype='band', output='sos')

    # Apply filter to extract F1 band
    audio_f1_band = signal.sosfilt(sos, audio)

    # Find peak in F1 band
    f1_peak = np.abs(audio_f1_band).max()

    # Normalize by F1 peak (with fallback)
    if f1_peak > 1e-8:
        return audio / f1_peak, f1_peak
    else:
        # Fallback to standard peak normalization
        peak = np.abs(audio).max()
        if peak > 1e-8:
            return audio / peak, peak
        return audio, 1.0


def extract_harmonic_spectrum(audio, f1_values, sr, n_harmonics=25, bandwidth_ratio=0.15, normalize_first=True):
    """
    Extract the amplitude of each harmonic from the audio.

    IMPORTANT: If normalize_first=True, the audio is F1-normalized before extracting harmonics.
    This ensures the prior matches what the model sees during training.

    Args:
        audio: Audio signal
        f1_values: F1 frequency for each frame
        sr: Sample rate
        n_harmonics: Number of harmonics to extract
        bandwidth_ratio: Bandwidth around each harmonic (±15%)
        normalize_first: If True, F1-normalize the audio first (CRITICAL for matching training data)

    Returns:
        harmonic_amplitudes: (n_harmonics,) array of mean amplitudes
    """
    # Get median F1 (ignoring zeros)
    f1_nonzero = f1_values[f1_values > 0]
    if len(f1_nonzero) == 0:
        return np.zeros(n_harmonics)

    # CRITICAL: Normalize by F1 peak first (same as training data)
    if normalize_first:
        audio, _ = normalize_by_f1_peak(audio, f1_values, sr, bandwidth_ratio=0.3)

    median_f1 = np.median(f1_nonzero)
    f0 = median_f1 / 2.0  # F0 is half of F1

    harmonic_amplitudes = np.zeros(n_harmonics)

    for h in range(n_harmonics):
        # Harmonic frequency: F0 * (h+1)
        # h=0 -> F0, h=1 -> F1 (2*F0), h=2 -> F2 (3*F0), etc.
        harmonic_freq = f0 * (h + 1)

        # Define bandpass filter around this harmonic
        low_cutoff = harmonic_freq * (1 - bandwidth_ratio)
        high_cutoff = harmonic_freq * (1 + bandwidth_ratio)

        # Ensure valid cutoffs
        low_cutoff = max(5, low_cutoff)
        high_cutoff = min(sr / 2 - 10, high_cutoff)

        if low_cutoff >= high_cutoff:
            continue

        # Design bandpass filter
        nyquist = sr / 2
        try:
            sos = signal.butter(4, [low_cutoff / nyquist, high_cutoff / nyquist],
                               btype='band', output='sos')
            # Apply filter
            audio_band = signal.sosfilt(sos, audio)
            # Store RMS amplitude
            harmonic_amplitudes[h] = np.sqrt(np.mean(audio_band**2))
        except:
            # Filter design failed (probably cutoffs too close to 0 or Nyquist)
            harmonic_amplitudes[h] = 0.0

    return harmonic_amplitudes


def main():
    parser = argparse.ArgumentParser(
        description='Compute mean harmonic distribution from F1-normalized dataset\n'
                    'IMPORTANT: Audio is F1-normalized before computing harmonics to match training data!'
    )
    parser.add_argument('--data_dir', type=str, default='train_subset',
                       help='Directory containing audio files')
    parser.add_argument('--f0_dir', type=str, default='train_subset/f0_corrected',
                       help='Directory containing F1 CSV files')
    parser.add_argument('--n_harmonics', type=int, default=25,
                       help='Number of harmonics to compute')
    parser.add_argument('--output', type=str, default='configs/harmonic_prior.npy',
                       help='Output file for harmonic prior')
    parser.add_argument('--plot', action='store_true',
                       help='Plot the resulting prior')
    parser.add_argument('--no_normalize', action='store_true',
                       help='DO NOT F1-normalize before computing (not recommended - won\'t match training)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    f0_dir = Path(args.f0_dir)

    # Get all audio files
    audio_files = sorted(list(data_dir.glob("*.wav")))
    print(f"Found {len(audio_files)} audio files")

    normalize_first = not args.no_normalize
    if normalize_first:
        print("IMPORTANT: Audio will be F1-normalized before computing harmonics (matches training data)")
    else:
        print("WARNING: Computing harmonics from non-normalized audio (won't match training data!)")

    # Collect harmonic distributions
    all_harmonic_distributions = []

    for audio_file in tqdm(audio_files, desc="Computing harmonic spectra"):
        # Load audio
        audio, sr = sf.read(audio_file)

        # Load F1 values
        f0_file = f0_dir / f"{audio_file.stem}.f0.csv"
        if not f0_file.exists():
            print(f"Warning: Missing F0 file for {audio_file.name}, skipping")
            continue

        f0_data = pd.read_csv(f0_file)
        f1_values = f0_data["frequency"].values  # These are F1 values

        # Extract harmonic spectrum (with F1 normalization to match training data)
        harmonic_amps = extract_harmonic_spectrum(
            audio, f1_values, sr, args.n_harmonics, normalize_first=normalize_first
        )

        # Normalize by sum (convert to probability distribution)
        total = harmonic_amps.sum()
        if total > 1e-8:
            harmonic_dist = harmonic_amps / total
            all_harmonic_distributions.append(harmonic_dist)

    # Compute mean harmonic distribution
    if len(all_harmonic_distributions) == 0:
        print("\nERROR: No files were successfully processed!")
        print(f"  Data directory: {data_dir.absolute()}")
        print(f"  F0 directory: {f0_dir.absolute()}")
        print(f"  Found {len(audio_files)} audio files")
        print("\nPlease ensure:")
        print("  1. Audio files exist in the data directory")
        print("  2. Corresponding .f0.csv files exist in the f0 directory")
        print("  3. The f0 directory path is correct")
        return

    all_distributions = np.array(all_harmonic_distributions)
    mean_distribution = np.mean(all_distributions, axis=0)
    std_distribution = np.std(all_distributions, axis=0)

    print(f"\nProcessed {len(all_harmonic_distributions)} files")
    print("\nMean harmonic distribution:")
    for i in range(min(10, args.n_harmonics)):
        freq_name = f"F{i}" if i == 0 else f"F{i}"
        approx_freq = f"(~{(i+1)*20} Hz)" if i < 5 else ""
        print(f"  {freq_name:3s} {approx_freq:12s}: {mean_distribution[i]:.4f} ± {std_distribution[i]:.4f}")

    # Save
    np.save(args.output, mean_distribution)
    print(f"\nSaved harmonic prior to: {args.output}")

    # Plot if requested
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot mean with std
        x = np.arange(args.n_harmonics)
        ax1.bar(x, mean_distribution, yerr=std_distribution, capsize=3, alpha=0.7, color='blue')
        ax1.set_xlabel('Harmonic Number')
        ax1.set_ylabel('Mean Amplitude (normalized)')
        title_suffix = " (F1-Normalized)" if normalize_first else " (Non-Normalized)"
        ax1.set_title(f'Mean Harmonic Distribution Across Dataset{title_suffix}')
        ax1.grid(True, alpha=0.3)

        # Add harmonic labels
        labels = [f'F{i}' for i in range(args.n_harmonics)]
        ax1.set_xticks(x[::2])  # Show every other label
        ax1.set_xticklabels(labels[::2])

        # Plot individual samples (first 50)
        for i, dist in enumerate(all_distributions[:50]):
            ax2.plot(x, dist, alpha=0.1, color='gray')
        ax2.plot(x, mean_distribution, linewidth=2, color='red', label='Mean')
        ax2.set_xlabel('Harmonic Number')
        ax2.set_ylabel('Amplitude (normalized)')
        ax2.set_title('Individual Sample Distributions (first 50) + Mean (red)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plot_path = args.output.replace('.npy', '.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to: {plot_path}")
        plt.show()


if __name__ == "__main__":
    main()
