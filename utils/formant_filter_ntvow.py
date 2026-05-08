"""
Apply adaptive formant filtering to NTVOW vowels to make them resemble elephant rumbles.

Processing:
1. Extract F1 (first formant) from spectrum
2. For each frame, apply adaptive filter:
   - Notch at F0 (suppress fundamental)
   - Pole/resonance at F1 (enhance first formant)
   - Decay towards 750 Hz (suppress higher frequencies)

This makes vowels more rumble-like by emphasizing F1 over F0.
"""

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from tqdm import tqdm
import librosa


def apply_formant_filtering(audio, f0_contour, sr,
                           f1_multiplier=2.0,
                           f0_notch_q=3.0,
                           f1_pole_q=2.5,
                           f1_boost_db=6.0,
                           decay_cutoff=750,
                           decay_steepness=1.5):
    """
    Apply time-variant formant filtering to audio.

    Args:
        audio: Input audio
        f0_contour: F0 contour from f0_corrected (frame rate, not corrected)
        sr: Sample rate
        f1_multiplier: Multiplier to estimate F1 from F0
        f0_notch_q: Q factor for F0 notch filter (higher = narrower)
        f1_pole_q: Q factor for F1 resonance (higher = narrower peak)
        f1_boost_db: Amount to boost F1 (dB)
        decay_cutoff: Frequency above which to start decay
        decay_steepness: Steepness of decay (higher = faster rolloff)

    Returns:
        filtered_audio: Filtered audio signal
        mean_f1_frequency: The mean F1 frequency used
    """
    # Use STFT for frame-wise filtering
    hop_length = int(sr * 0.016)  # 16ms frames
    n_fft = 4096

    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Number of STFT frames
    n_frames = D.shape[1]

    # Match F0 contour to STFT frames (handle potential length mismatch)
    if len(f0_contour) != n_frames:
        # Resample f0_contour to match STFT frames
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(f0_contour))
        x_new = np.linspace(0, 1, n_frames)
        f0_interp = interp1d(x_old, f0_contour, kind='linear', fill_value='extrapolate')
        f0_per_frame = f0_interp(x_new)
    else:
        f0_per_frame = f0_contour.copy()

    # Compute mean F0 and F1 for reporting
    valid_f0 = f0_per_frame[f0_per_frame > 0]
    if len(valid_f0) > 0:
        mean_f0 = np.mean(valid_f0)
        mean_f1 = mean_f0 * f1_multiplier
        print(f"  Mean F0: {mean_f0:.1f} Hz → Mean F1 estimate: {mean_f1:.1f} Hz")
        print(f"  F0 range: {np.min(valid_f0):.1f} - {np.max(valid_f0):.1f} Hz")
    else:
        print("  Warning: No valid F0 found, using defaults")
        mean_f1 = 50.0

    # Pre-compute boost factor
    f1_boost_linear = 10 ** (f1_boost_db / 20)

    # Initialize output spectrogram
    filtered_D = np.zeros_like(D)

    # Apply time-variant filtering: for each frame, compute filter based on F0 at that time
    for frame_idx in range(n_frames):
        f0_value = f0_per_frame[frame_idx]

        # Use default F0 if unvoiced
        if f0_value <= 0:
            f0_value = 25.0  # Default F0

        # Estimate F1 from F0 for this frame
        f1_value = f0_value * f1_multiplier

        # Build frame-specific filter
        filter_response = np.ones_like(freqs)

        # 1. Notch at F0 (suppress fundamental)
        f0_bw = f0_value / f0_notch_q
        f0_notch = 1.0 - np.exp(-((freqs - f0_value) / f0_bw) ** 2)
        filter_response *= f0_notch

        # 2. Resonance/pole at F1 (boost first formant)
        f1_bw = f1_value / f1_pole_q
        f1_resonance = 1.0 + (f1_boost_linear - 1.0) * np.exp(-((freqs - f1_value) / f1_bw) ** 2)
        filter_response *= f1_resonance

        # 3. Smooth taper starting from F1
        taper_start = f1_value
        taper_filter = 1.0 / (1.0 + ((freqs - taper_start) / (taper_start * 2)) ** (2 * decay_steepness))
        taper_filter = np.where(freqs > taper_start, taper_filter, 1.0)
        filter_response *= taper_filter

        # 4. Final hard cutoff at decay_cutoff
        final_decay = 1.0 / (1.0 + (freqs / decay_cutoff) ** (2 * decay_steepness))
        filter_response *= final_decay

        # Apply filter to this frame
        filtered_D[:, frame_idx] = D[:, frame_idx] * filter_response

    # Inverse STFT
    filtered_audio = librosa.istft(filtered_D, hop_length=hop_length, length=len(audio))

    return filtered_audio, mean_f1


def process_dataset(input_dir, output_dir,
                   f1_multiplier=2.0,
                   f0_notch_q=3.0,
                   f1_pole_q=2.5,
                   f1_boost_db=6.0,
                   decay_cutoff=750,
                   decay_steepness=1.5,
                   skip_existing=True):
    """Process entire dataset with formant filtering."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find F0 directory
    f0_dir = input_dir / "f0_corrected"
    if not f0_dir.exists():
        raise ValueError(f"F0 directory not found: {f0_dir}")

    # Find all audio files
    audio_files = sorted(list(input_dir.glob("*.wav")))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {input_dir}")

    print("="*60)
    print("FORMANT FILTERING FOR NTVOW VOWELS")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"F0 directory: {f0_dir}")
    print(f"Files to process: {len(audio_files)}")
    print(f"Parameters:")
    print(f"  F1 multiplier: {f1_multiplier}")
    print(f"  F0 notch Q: {f0_notch_q} (higher = narrower notch)")
    print(f"  F1 pole Q: {f1_pole_q} (higher = narrower peak)")
    print(f"  F1 boost: {f1_boost_db} dB")
    print(f"  Decay cutoff: {decay_cutoff} Hz")
    print(f"  Decay steepness: {decay_steepness} (lower = slower taper)")
    print(f"Skip existing: {skip_existing}")
    print()

    processed = 0
    skipped = 0
    errors = 0

    for audio_path in tqdm(audio_files, desc="Processing"):
        output_path = output_dir / audio_path.name

        if skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            # Load audio
            audio, sr = sf.read(audio_path)

            # Load F0
            f0_path = f0_dir / f"{audio_path.stem}.f0.csv"
            if not f0_path.exists():
                print(f"\n  Warning: F0 file not found for {audio_path.name}, skipping")
                skipped += 1
                continue

            df = pd.read_csv(f0_path)
            f0_contour = df['frequency'].values / 2

            # Apply formant filtering
            filtered_audio, f1_frequency = apply_formant_filtering(
                audio, f0_contour, sr,
                f1_multiplier=f1_multiplier,
                f0_notch_q=f0_notch_q,
                f1_pole_q=f1_pole_q,
                f1_boost_db=f1_boost_db,
                decay_cutoff=decay_cutoff,
                decay_steepness=decay_steepness
            )

            # Save filtered audio
            sf.write(output_path, filtered_audio, sr)

            processed += 1

        except Exception as e:
            print(f"\n  Error processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
            continue

    print()
    print("="*60)
    print("✓ Formant filtering complete!")
    print("="*60)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print()
    print(f"Filtered audio saved to: {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Listen to a few samples to verify filtering")
    print(f"  2. Use filtered audio for training (F0 already extracted)")
    print(f"  3. Copy F0 files from input to output if needed:"
          f"\n     cp -r {input_dir}/f0_corrected {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Apply formant filtering to NTVOW vowels to make them more rumble-like',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python preprocessing/formant_filter_ntvow.py --input data/ntvow_low --output data/ntvow_low_filtered

What it does:
  1. Loads F0 from f0_corrected/ directory
  2. Estimates F1 as f1_multiplier × mean(F0) (default: 2× mean F0)
  3. Applies fixed filter to entire file:
     - Notch at F0 (suppress fundamental)
     - Resonance/pole at F1 (boost estimated formant)
     - Smooth taper starting from F1 (gradually suppress higher frequencies)
     - Final cutoff above 750 Hz (hard suppression)

This makes vowels more similar to elephant rumbles where F1 is dominant over F0.
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (should contain audio + f0_corrected/)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for filtered audio')
    parser.add_argument('--f1_multiplier', type=float, default=2.0,
                        help='F1 multiplier (default: 2.0)')
    parser.add_argument('--f0_notch_q', type=float, default=3.0,
                        help='F0 notch Q factor (default: 3.0, lower = wider notch)')
    parser.add_argument('--f1_pole_q', type=float, default=0.1,
                        help='F1 pole Q factor (default: 2.5, lower = broader peak)')
    parser.add_argument('--f1_boost_db', type=float, default=0.0,
                        help='F1 boost in dB (default: 6.0)')
    parser.add_argument('--decay_cutoff', type=float, default=8000,
                        help='Decay cutoff frequency (default: 750 Hz)')
    parser.add_argument('--decay_steepness', type=float, default=0.1,
                        help='Decay steepness (default: 1.5, slower rolloff)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip files that already exist (default: True)')
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false',
                        help='Reprocess all files')

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        f1_multiplier=args.f1_multiplier,
        f0_notch_q=args.f0_notch_q,
        f1_pole_q=args.f1_pole_q,
        f1_boost_db=args.f1_boost_db,
        decay_cutoff=args.decay_cutoff,
        decay_steepness=args.decay_steepness,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
