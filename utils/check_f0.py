"""
Visualize f0 extraction quality
Plots spectrograms (0-100 Hz) with extracted f0 overlay
"""

import argparse
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm
import soundfile as sf


def plot_f0_on_spectrogram(audio_path, f0_path, sr=16000, n_fft=8192,
                           fmin=0, fmax=100, output_path=None):
    """
    Plot spectrogram with f0 overlay for a single file

    Args:
        audio_path: Path to audio file
        f0_path: Path to f0 CSV file
        sr: Sampling rate
        n_fft: FFT size (large for good low-freq resolution)
        fmin: Min frequency to display
        fmax: Max frequency to display
        output_path: Optional path to save figure
    """
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)

    # Load f0 data
    df = pd.read_csv(f0_path)
    f0_time = df['time'].values
    f0_freq = df['frequency'].values
    f0_conf = df['confidence'].values

    # Compute spectrogram with high frequency resolution
    hop_length = 256  # ~16ms at 16kHz
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Time and frequency axes
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # Plot 1: Full spectrogram (0-100 Hz)
    ax = axes[0]
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    im = ax.pcolormesh(times, freqs[freq_mask], S_db[freq_mask, :],
                       shading='gouraud', cmap='magma', vmin=-80, vmax=0)

    # Overlay f0 (only confident frames)
    confident_mask = f0_conf > 0.01
    ax.plot(f0_time[confident_mask], f0_freq[confident_mask],
            'c-', linewidth=2, label='Extracted f0', alpha=0.8)
    ax.plot(f0_time[~confident_mask], f0_freq[~confident_mask],
            'c:', linewidth=1, alpha=0.5)

    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(f'Spectrogram + Extracted F0\n{audio_path.name}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([fmin, fmax])
    plt.colorbar(im, ax=ax, label='dB')

    # Plot 2: Zoomed to f0 region (f0 ± 20 Hz)
    ax = axes[1]
    valid_f0 = f0_freq[f0_freq > 0]
    if len(valid_f0) > 0:
        f0_median = np.median(valid_f0)
        zoom_fmin = max(0, f0_median - 20)
        zoom_fmax = min(fmax, f0_median + 20)

        zoom_mask = (freqs >= zoom_fmin) & (freqs <= zoom_fmax)
        im2 = ax.pcolormesh(times, freqs[zoom_mask], S_db[zoom_mask, :],
                           shading='gouraud', cmap='magma', vmin=-80, vmax=0)

        # Overlay f0
        ax.plot(f0_time[confident_mask], f0_freq[confident_mask],
                'c-', linewidth=2, label='Extracted f0', alpha=0.8)
        ax.plot(f0_time[~confident_mask], f0_freq[~confident_mask],
                'c:', linewidth=1, alpha=0.5)

        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'Zoomed View (f0 ± 20 Hz)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([zoom_fmin, zoom_fmax])
        plt.colorbar(im2, ax=ax, label='dB')

    # Plot 3: F0 contour and confidence
    ax = axes[2]
    ax2 = ax.twinx()

    # Plot f0
    ax.plot(f0_time, f0_freq, 'b-', linewidth=2, label='F0 (Hz)')
    ax.set_ylabel('F0 (Hz)', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_ylim([0, fmax])
    ax.grid(True, alpha=0.3)

    # Plot confidence
    ax2.plot(f0_time, f0_conf, 'r-', linewidth=1, alpha=0.7, label='Confidence')
    ax2.axhline(0.3, color='orange', linestyle='--', linewidth=1,
                alpha=0.5, label='Threshold (0.3)')
    ax2.set_ylabel('Confidence', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim([0, 1])

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title('F0 Contour and Confidence', fontsize=12, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def check_multiple_files(data_dir, n_samples=10, frame_resolution=0.016,
                         sr=16000, output_dir=None, random_seed=42):
    """
    Check f0 extraction quality for multiple files

    Args:
        data_dir: Directory with audio and f0 files
        n_samples: Number of samples to check
        frame_resolution: Frame resolution to find f0 directory
        sr: Sampling rate
        output_dir: Directory to save plots (optional)
        random_seed: Random seed for sampling
    """
    data_dir = Path(data_dir)
    f0_dir = data_dir / f"f0_{frame_resolution:.3f}"

    # Find audio files
    audio_files = sorted(list(data_dir.glob("*.wav")))

    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {data_dir}")
        return

    if not f0_dir.exists():
        print(f"ERROR: F0 directory not found: {f0_dir}")
        return

    print(f"\nFound {len(audio_files)} audio files")
    print(f"F0 directory: {f0_dir}")

    # Sample files
    # take the first n_samples files for consistency

    np.random.seed(random_seed)
    if len(audio_files) > n_samples:
        #sample_indices = np.random.choice(len(audio_files), n_samples, replace=False)
        sample_indices = np.arange(n_samples)
        sample_files = [audio_files[i] for i in sorted(sample_indices)]
    else:
        sample_files = audio_files[:n_samples]

    print(f"\nChecking {len(sample_files)} samples...")

    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # Process each file
    for i, audio_path in enumerate(sample_files, 1):
        print(f"\n[{i}/{len(sample_files)}] {audio_path.name}")

        # Find corresponding f0 file
        f0_path = f0_dir / f"{audio_path.stem}.f0.csv"

        if not f0_path.exists():
            print(f"  WARNING: F0 file not found: {f0_path.name}")
            continue

        # Load f0 data for statistics
        df = pd.read_csv(f0_path)
        f0_freq = df['frequency'].values
        f0_conf = df['confidence'].values

        # Statistics
        valid_f0 = f0_freq[f0_freq > 0]
        confident = (f0_conf > 0.3).sum() / len(f0_conf) * 100

        if len(valid_f0) > 0:
            print(f"  F0 range: {valid_f0.min():.1f} - {valid_f0.max():.1f} Hz")
            print(f"  F0 median: {np.median(valid_f0):.1f} Hz")
            print(f"  Confident frames: {confident:.1f}%")
        else:
            print(f"  WARNING: No voiced frames detected!")

        # Plot
        if output_dir:
            output_path = output_dir / f"f0_check_{audio_path.stem}.png"
        else:
            output_path = None

        fig = plot_f0_on_spectrogram(
            audio_path, f0_path, sr=sr,
            n_fft=8192, fmin=0, fmax=100,
            output_path=output_path
        )

        # Show interactively if not saving
        if not output_dir:
            plt.show()
        else:
            plt.close(fig)

    if output_dir:
        print(f"\n✓ All plots saved to: {output_dir}")

    print("\n" + "="*60)
    print("F0 EXTRACTION CHECK COMPLETE")
    print("="*60)
    print("\nWhat to look for:")
    print("  ✓ F0 contour (cyan line) should follow the brightest harmonic")
    print("  ✓ No octave errors (f0 jumping between harmonics)")
    print("  ✓ Confidence >0.3 for most voiced regions")
    print("  ✓ F0 should be 0 during unvoiced/silent regions")
    print("\nIf issues found:")
    print("  1. Adjust --fmin/--fmax in preprocess_elephant.py")
    print("  2. Increase median filter kernel_size")
    print("  3. Check input audio quality (noise, clipping)")


def main():
    parser = argparse.ArgumentParser(
        description='Check f0 extraction quality by visualizing spectrograms with f0 overlay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check 6 random samples, display interactively
  python check_f0_extraction.py --input data/rumbles

  # Check 10 samples and save plots
  python check_f0_extraction.py --input data/rumbles --n_samples 10 --output f0_check_plots

  # Check specific sampling rate and frame resolution
  python check_f0_extraction.py --input data/rumbles --sr 16000 --frame_resolution 0.016
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with preprocessed data (*.wav + f0_X.XXX/)')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to check (default: 10)')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution to find f0 directory (default: 0.016)')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate (default: 16000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory to save plots (default: show interactively)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')

    args = parser.parse_args()

    check_multiple_files(
        data_dir=args.input,
        n_samples=args.n_samples,
        frame_resolution=args.frame_resolution,
        sr=args.sr,
        output_dir=args.output,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
