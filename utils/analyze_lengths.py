"""
Analyze audio dataset to determine optimal waveform_sec parameter
Checks distribution of audio lengths and spectral content
"""

import argparse
import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal


def compute_average_spectrum(audio_files, sr=16000, n_fft=4096, max_files=50):
    """
    Compute average power spectral density across dataset

    Args:
        audio_files: List of audio file paths
        sr: Sampling rate
        n_fft: FFT size
        max_files: Maximum number of files to analyze (for speed)

    Returns:
        freqs: Frequency bins
        avg_psd: Average power spectral density
    """
    print(f"\nComputing spectral distribution from {min(len(audio_files), max_files)} files...")

    # Sample files if too many
    if len(audio_files) > max_files:
        sample_indices = np.random.choice(len(audio_files), max_files, replace=False)
        sampled_files = [audio_files[i] for i in sample_indices]
    else:
        sampled_files = audio_files

    all_psds = []

    for audio_file in tqdm(sampled_files, desc="Computing spectra"):
        try:
            y, _ = librosa.load(audio_file, sr=sr)

            # Compute power spectral density
            freqs, psd = signal.welch(y, sr, nperseg=n_fft, scaling='density')
            all_psds.append(psd)

        except Exception as e:
            continue

    if len(all_psds) == 0:
        return None, None

    # Average across all files (in log domain for better representation)
    all_psds = np.array(all_psds)
    avg_psd = np.mean(all_psds, axis=0)

    return freqs, avg_psd


def analyze_dataset_lengths(audio_dir, sr=16000, plot=True, extensions=None, analyze_spectrum=True):
    """
    Analyze distribution of audio lengths and spectral content in dataset

    Args:
        audio_dir: Directory containing audio files
        sr: Sampling rate (for loading audio)
        plot: Whether to show visualization
        extensions: List of audio extensions to include
        analyze_spectrum: Whether to compute spectral distribution

    Returns:
        durations: Array of durations in seconds
    """
    if extensions is None:
        extensions = ['*.wav', '*.WAV', '*.mp3', '*.flac', '*.ogg', '*.m4a']

    audio_dir = Path(audio_dir)

    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(audio_dir.glob(ext)))
        # Also search subdirectories
        audio_files.extend(list(audio_dir.glob(f'**/{ext}')))

    # Remove duplicates
    audio_files = list(set(audio_files))

    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {audio_dir}")
        print(f"Searched for extensions: {extensions}")
        return None

    print(f"\nAnalyzing {len(audio_files)} audio files from {audio_dir}")
    print(f"Extensions found: {set([f.suffix for f in audio_files])}\n")

    # Load each file and get duration
    durations = []
    failed_files = []

    for audio_file in tqdm(audio_files, desc="Loading audio files"):
        try:
            y, _ = librosa.load(audio_file, sr=sr)
            duration = len(y) / sr
            durations.append(duration)
        except Exception as e:
            print(f"\nWarning: Could not load {audio_file.name}: {e}")
            failed_files.append(audio_file)
            continue

    if len(durations) == 0:
        print("ERROR: No audio files could be loaded successfully")
        return None

    durations = np.array(durations)

    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total files: {len(durations)}")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
    print(f"\nDuration range: {durations.min():.2f} - {durations.max():.2f} seconds")
    print(f"Mean: {durations.mean():.2f}s")
    print(f"Median: {np.median(durations):.2f}s")
    print(f"Std dev: {durations.std():.2f}s")

    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}%: {np.percentile(durations, p):6.2f}s")

    # Check usability for different waveform_sec values
    print("\n" + "="*60)
    print("USABILITY FOR DIFFERENT TRAINING WINDOW SIZES")
    print("="*60)

    test_values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
    best_waveform_sec = None
    best_coverage = 0

    for waveform_sec in test_values:
        usable = (durations >= waveform_sec).sum()
        percentage = usable / len(durations) * 100
        marker = ""

        # Find sweet spot (>90% coverage)
        if percentage >= 90 and (best_waveform_sec is None or waveform_sec > best_waveform_sec):
            best_waveform_sec = waveform_sec
            best_coverage = percentage
            marker = " ← Recommended"

        print(f"waveform_sec={waveform_sec:3.1f}s: {usable:3d}/{len(durations)} files "
              f"({percentage:5.1f}% usable){marker}")

    # Compute spectral distribution
    freqs, avg_psd = None, None
    if analyze_spectrum:
        freqs, avg_psd = compute_average_spectrum(audio_files, sr=sr)

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if best_waveform_sec:
        print(f"\n✓ Recommended waveform_sec: {best_waveform_sec}s")
        print(f"  - Uses {best_coverage:.1f}% of your data")
        print(f"  - Provides good temporal context for elephant rumbles")
    else:
        # Find value with 80%+ coverage
        for waveform_sec in reversed(test_values):
            usable = (durations >= waveform_sec).sum()
            percentage = usable / len(durations) * 100
            if percentage >= 80:
                print(f"\n⚠ Recommended waveform_sec: {waveform_sec}s")
                print(f"  - Uses {percentage:.1f}% of your data")
                print(f"  - Consider padding approach for short files")
                break

    # Spectral analysis recommendations
    if freqs is not None and avg_psd is not None:
        # Find where most energy is concentrated
        psd_db = 10 * np.log10(avg_psd + 1e-10)
        max_energy_idx = np.argmax(psd_db)
        max_energy_freq = freqs[max_energy_idx]

        # Find frequency range containing 90% of energy
        cumulative_energy = np.cumsum(avg_psd) / np.sum(avg_psd)
        f90_idx = np.where(cumulative_energy >= 0.90)[0][0]
        f90 = freqs[f90_idx]

        print("\n" + "="*60)
        print("SPECTRAL ANALYSIS")
        print("="*60)
        print(f"Peak energy at: {max_energy_freq:.1f} Hz")
        print(f"90% of energy below: {f90:.1f} Hz")

        if f90 < sr/4:
            print(f"✓ Good: Most energy is in lower frequencies (<{sr/4:.0f} Hz)")
            print(f"  - Sampling rate of {sr} Hz is appropriate")

        if max_energy_freq < 50:
            print(f"✓ Low-frequency dominated: confirms elephant rumble characteristics")
            print(f"  - n_fft=4096 in config is appropriate for this frequency range")

    # Additional advice based on distribution
    short_files = (durations < 2).sum()
    if short_files > len(durations) * 0.2:
        print(f"\n⚠ Warning: {short_files} files ({short_files/len(durations)*100:.1f}%) are <2 seconds")
        print("  Consider:")
        print("  1. Using waveform_sec=1.5 or 2.0")
        print("  2. Implementing padding for very short files")
        print("  3. Filtering out very short files if not critical")

    very_long = (durations > 10).sum()
    if very_long > 0:
        print(f"\n✓ Good: {very_long} files ({very_long/len(durations)*100:.1f}%) are >10 seconds")
        print("  - Longer files provide more training diversity via random cropping")

    # Files that would be excluded with different settings
    current_setting = 4  # Default from config
    excluded = len(durations) - (durations >= current_setting).sum()
    if excluded > 0:
        print(f"\n⚠ With current config (waveform_sec={current_setting}s):")
        print(f"  - {excluded} files will be excluded ({excluded/len(durations)*100:.1f}%)")
        print(f"  - Files excluded: <{current_setting}s duration")

    # Visualization
    if plot:
        n_plots = 3 if (freqs is not None and avg_psd is not None) else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5*n_plots))

        # Histogram
        axes[0].hist(durations, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(4, color='red', linestyle='--', linewidth=2,
                       label=f'Current config (waveform_sec=4s)')
        if best_waveform_sec and best_waveform_sec != 4:
            axes[0].axvline(best_waveform_sec, color='green', linestyle='--', linewidth=2,
                           label=f'Recommended (waveform_sec={best_waveform_sec}s)')
        axes[0].set_xlabel('Duration (seconds)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Audio File Lengths', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Cumulative distribution
        sorted_durations = np.sort(durations)
        cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100

        axes[1].plot(sorted_durations, cumulative, linewidth=2, color='steelblue')
        axes[1].axvline(4, color='red', linestyle='--', linewidth=2,
                       label=f'Current config (waveform_sec=4s)')
        axes[1].axhline(90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
        if best_waveform_sec and best_waveform_sec != 4:
            axes[1].axvline(best_waveform_sec, color='green', linestyle='--', linewidth=2,
                           label=f'Recommended (waveform_sec={best_waveform_sec}s)')
        axes[1].set_xlabel('Duration (seconds)', fontsize=12)
        axes[1].set_ylabel('Cumulative % of files', fontsize=12)
        axes[1].set_title('Cumulative Distribution (What % of files are usable?)',
                         fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 105])

        # Spectral distribution
        if freqs is not None and avg_psd is not None:
            psd_db = 10 * np.log10(avg_psd + 1e-10)

            axes[2].plot(freqs, psd_db, linewidth=2, color='darkblue')
            axes[2].axvline(50, color='red', linestyle='--', linewidth=1.5,
                           alpha=0.7, label='50 Hz (typical upper f0 limit)')
            axes[2].axvline(sr/2, color='gray', linestyle=':', linewidth=1.5,
                           alpha=0.5, label=f'Nyquist ({sr/2:.0f} Hz)')
            axes[2].set_xlabel('Frequency (Hz)', fontsize=12)
            axes[2].set_ylabel('Power Spectral Density (dB)', fontsize=12)
            axes[2].set_title('Average Spectral Distribution', fontsize=14, fontweight='bold')
            axes[2].set_xlim([0, min(2000, sr/2)])  # Focus on relevant range
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)

            # Add annotations for key frequency ranges
            axes[2].axvspan(5, 35, alpha=0.2, color='green', label='Typical elephant f0 (5-35 Hz)')
            axes[2].axvspan(35, 500, alpha=0.1, color='orange', label='Harmonic content')
            axes[2].legend(fontsize=9, loc='upper right')

        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: dataset_analysis.png")
        plt.show()

    return durations


def main():
    parser = argparse.ArgumentParser(
        description='Analyze audio dataset lengths and spectral content for DDSP training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze training data
  python analyze_dataset_lengths.py --input raw_data/elephant/train

  # Analyze both splits
  python analyze_dataset_lengths.py --input raw_data/elephant/train
  python analyze_dataset_lengths.py --input raw_data/elephant/test

  # Analyze before splitting
  python analyze_dataset_lengths.py --input raw_data/elephant_audio

  # Don't show plot (just print stats)
  python analyze_dataset_lengths.py --input raw_data/elephant/train --no-plot

  # Skip spectral analysis (faster)
  python analyze_dataset_lengths.py --input raw_data/elephant/train --no-spectrum
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate for loading audio (default: 16000)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not show visualization')
    parser.add_argument('--no-spectrum', action='store_true',
                        help='Skip spectral analysis (faster)')
    parser.add_argument('--extensions', nargs='+', default=None,
                        help='Audio extensions to include (default: wav, mp3, flac, ogg, m4a)')

    args = parser.parse_args()

    # Format extensions if provided
    if args.extensions:
        extensions = [f'*.{ext.lstrip("*.")}' for ext in args.extensions]
    else:
        extensions = None

    durations = analyze_dataset_lengths(
        audio_dir=args.input,
        sr=args.sr,
        plot=not args.no_plot,
        extensions=extensions,
        analyze_spectrum=not args.no_spectrum
    )

    if durations is not None:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
