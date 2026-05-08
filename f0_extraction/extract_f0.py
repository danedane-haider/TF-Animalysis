"""
Extract f0 from preprocessed 16kHz audio files
Allows testing different f0 extraction parameters without re-resamplingi
"""

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import argparse
import sys
sys.path.append(str(Path(__file__).parent))
from tf_transforms.transforms import Elelet


def highpass_filter(audio, sr, cutoff=10.0, order=5):
    """Remove DC offset and subsonic noise"""
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, audio)
    return filtered


def median_filter_pitch(frequency, confidence, kernel_size=9, conf_threshold=0.2):
    """Apply median filtering to remove octave errors"""
    mask = confidence > conf_threshold
    freq_filtered = frequency.copy()

    if mask.sum() > kernel_size:
        confident_freqs = frequency[mask]
        filtered = median_filter(confident_freqs, size=kernel_size)
        freq_filtered[mask] = filtered

    return freq_filtered


def extract_f0_pyin(audio, sr=16000, frame_resolution=0.016, fmin=5, fmax=50,
                    use_pitch_shift=False, pitch_shift_octaves=2, extract_f1=False):
    """
    Extract f0 using librosa's pYIN algorithm

    Args:
        audio: audio signal
        sr: sampling rate
        frame_resolution: seconds per frame
        fmin: minimum frequency (Hz)
        fmax: maximum frequency (Hz)
        use_pitch_shift: if True, pitch shift up before extraction for better accuracy
        pitch_shift_octaves: number of octaves to shift up (default: 2)
        extract_f1: if True, extract F1 (first overtone = 2×F0) then divide by 2
                    This is more robust when F1 is stronger than F0

    Returns:
        time, f0, confidence arrays
    """
    # Apply highpass filter
    audio_filtered = highpass_filter(audio, sr, cutoff=fmin * 0.5)

    hop_length = int(sr * frame_resolution)

    if extract_f1:
        # Extract F1 (first overtone at 2×F0) then divide by 2
        f1_min = fmin * 2
        f1_max = fmax * 2

        # For 10 Hz (F1 when F0=5Hz): 2 periods = 0.2s = 3200 samples at 16kHz
        # Use 4096 (next power of 2) for efficiency
        frame_length = 4096

        f1, voiced_flag, voiced_probs = librosa.pyin(
            audio_filtered,
            fmin=f1_min,
            fmax=f1_max,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            pad_mode='constant'
        )

        # Convert NaN to 0
        f1 = np.nan_to_num(f1, nan=0.0)

        # Divide by 2 to get F0 from F1
        f0 = f1 / 2.0

    else:
        # Original approach: direct pYIN on low frequencies
        # For accurate low-freq detection, need at least 2 periods of fmin in frame
        # For 5 Hz: 2 periods = 0.4s = 6400 samples at 16kHz
        # Use 8192 (next power of 2) for efficiency
        frame_length = 8192

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_filtered,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            pad_mode='constant'
        )

        # Convert NaN to 0
        f0 = np.nan_to_num(f0, nan=0.0)

    # Apply median filtering to remove octave errors
    f0 = median_filter_pitch(f0, voiced_probs, kernel_size=5, conf_threshold=0.1)

    # Enforce frequency range (use original fmin/fmax)
    f0[(f0 < fmin) | (f0 > fmax)] = 0.0

    # Create time array
    time = np.arange(len(f0)) * frame_resolution
    confidence = voiced_probs

    return time, f0, confidence


def extract_f0_elelet(audio, sr=16000, stride=256, fmin=10, fmax=50, divide_by_2=False, max_jump=5.0, use_global_peak=False, energy_threshold=0.3):
    """
    Extract f0 using Elelet spectrogram by finding peak in the specified frequency range.

    Args:
        audio: audio signal
        sr: sampling rate (default: 16000)
        stride: hop length for Elelet transform (default: 256)
        fmin: minimum frequency to search (default: 15 Hz)
        fmax: maximum frequency to search (default: 50 Hz)
        divide_by_2: if True, divide detected frequency by 2 (for F1->F0 conversion)
        max_jump: maximum allowed frequency jump between frames in Hz (default: 5.0)
                  Set to None to disable continuity constraint
        use_global_peak: if True, find global peak by summing across time, then track
                        with max_jump from strongest frame outward (default: False)
        energy_threshold: fraction of max peak energy for automatic start/end detection
                         (default: 0.3, meaning 30% of max peak)

    Returns:
        time, f0, confidence arrays
    """
    # Initialize Elelet transform
    transform = Elelet(
        kernel_size=16000+8000,
        num_channels=1024,
        stride=256,
        fmin=5,
        fmax=100,
        fs=16000,
        supp_mult=0.2,
        scale='elelog',
    )

    # Compute Elelet coefficients
    coeffs = transform(audio)
    coeffs_abs = np.abs(coeffs)

    # Get center frequencies
    fc = transform.fc
    if hasattr(fc, 'numpy'):
        fc = fc.numpy()

    # Align center frequencies to coefficient channels
    num_freq_bins = coeffs_abs.shape[0]
    fc_filtered = fc[:num_freq_bins]

    # Find frequency indices for the search range
    freq_mask = (fc_filtered >= fmin) & (fc_filtered <= fmax)
    freq_indices = np.where(freq_mask)[0]
    fc_search = fc_filtered[freq_mask]

    num_frames = coeffs_abs.shape[1]
    f0 = np.zeros(num_frames)
    confidence = np.zeros(num_frames)

    if use_global_peak:
        # Two-stage approach:
        # 1. Find global dominant frequency by summing across time
        # 2. Track from center outward with strict 1Hz continuity

        # Sum TF coefficients along time within search range
        time_avg = np.mean(coeffs_abs[freq_mask, :], axis=1)

        # Find peak in the averaged spectrum
        global_peak_idx = np.argmax(time_avg)
        global_peak_freq = fc_search[global_peak_idx]

        print(f"  Peak frequency: {global_peak_freq:.2f} Hz")

        # Find the frame with strongest energy in the search range (best starting point)
        frame_energies = np.sum(coeffs_abs[freq_mask, :], axis=0)
        start_frame = np.argmax(frame_energies)

        # Initialize starting frame with global peak
        frame_coeffs = coeffs_abs[freq_mask, start_frame]
        # Find peak closest to global peak
        jump_threshold = max_jump if max_jump is not None else 1
        allowed_mask = np.abs(fc_search - global_peak_freq) <= jump_threshold
        if np.any(allowed_mask):
            allowed_coeffs = frame_coeffs.copy()
            allowed_coeffs[~allowed_mask] = -np.inf
            peak_idx = np.argmax(allowed_coeffs)
        else:
            peak_idx = global_peak_idx

        peak_freq = fc_search[peak_idx]
        peak_magnitude = frame_coeffs[peak_idx]

        f0[start_frame] = peak_freq / 2.0 if divide_by_2 else peak_freq
        frame_energy = np.mean(frame_coeffs)
        confidence[start_frame] = peak_magnitude / (frame_energy * len(frame_coeffs)) if frame_energy > 0 else 0.0

        # Track forward (right) from start frame
        for frame_idx in range(start_frame + 1, num_frames):
            frame_coeffs = coeffs_abs[freq_mask, frame_idx]
            prev_freq = f0[frame_idx - 1] * (2.0 if divide_by_2 else 1.0)  # Convert back to pre-division freq

            # Apply continuity constraint
            allowed_mask = np.abs(fc_search - prev_freq) <= jump_threshold

            if np.any(allowed_mask):
                allowed_coeffs = frame_coeffs.copy()
                allowed_coeffs[~allowed_mask] = -np.inf
                peak_idx = np.argmax(allowed_coeffs)
            else:
                peak_idx = np.argmax(frame_coeffs)

            peak_freq = fc_search[peak_idx]
            peak_magnitude = frame_coeffs[peak_idx]

            f0[frame_idx] = peak_freq / 2.0 if divide_by_2 else peak_freq
            frame_energy = np.mean(frame_coeffs)
            confidence[frame_idx] = peak_magnitude / (frame_energy * len(frame_coeffs)) if frame_energy > 0 else 0.0

        # Track backward (left) from start frame
        for frame_idx in range(start_frame - 1, -1, -1):
            frame_coeffs = coeffs_abs[freq_mask, frame_idx]
            prev_freq = f0[frame_idx + 1] * (2.0 if divide_by_2 else 1.0)  # Convert back to pre-division freq

            # Apply continuity constraint
            allowed_mask = np.abs(fc_search - prev_freq) <= jump_threshold

            if np.any(allowed_mask):
                allowed_coeffs = frame_coeffs.copy()
                allowed_coeffs[~allowed_mask] = -np.inf
                peak_idx = np.argmax(allowed_coeffs)
            else:
                peak_idx = np.argmax(frame_coeffs)

            peak_freq = fc_search[peak_idx]
            peak_magnitude = frame_coeffs[peak_idx]

            f0[frame_idx] = peak_freq / 2.0 if divide_by_2 else peak_freq
            frame_energy = np.mean(frame_coeffs)
            confidence[frame_idx] = peak_magnitude / (frame_energy * len(frame_coeffs)) if frame_energy > 0 else 0.0

    else:
        # Original approach: frame-by-frame with optional continuity constraint
        # For each time frame, find the peak frequency in the search range
        for frame_idx in range(num_frames):
            frame_coeffs = coeffs_abs[freq_mask, frame_idx]

            if max_jump is not None and frame_idx > 0 and f0[frame_idx - 1] > 0:
                # Apply continuity constraint: prefer peaks near previous frequency
                prev_freq = f0[frame_idx - 1]

                # Find all peaks within max_jump of previous frequency
                allowed_mask = np.abs(fc_search - prev_freq) <= max_jump

                if np.any(allowed_mask):
                    # Find strongest peak within allowed range
                    allowed_coeffs = frame_coeffs.copy()
                    allowed_coeffs[~allowed_mask] = -np.inf  # Exclude disallowed peaks
                    peak_idx = np.argmax(allowed_coeffs)
                else:
                    # No peaks within range, fall back to global maximum
                    peak_idx = np.argmax(frame_coeffs)
            else:
                # First frame or no continuity constraint
                peak_idx = np.argmax(frame_coeffs)

            peak_freq = fc_search[peak_idx]
            peak_magnitude = frame_coeffs[peak_idx]

            # Store detected frequency (optionally divide by 2 for F1->F0 conversion)
            if divide_by_2:
                f0[frame_idx] = peak_freq / 2.0
            else:
                f0[frame_idx] = peak_freq

            # Confidence: normalized peak magnitude
            # Higher magnitude = higher confidence
            frame_energy = np.mean(frame_coeffs)
            if frame_energy > 0:
                confidence[frame_idx] = peak_magnitude / (frame_energy * len(frame_coeffs))
            else:
                confidence[frame_idx] = 0.0

    # Create time array
    time = np.arange(num_frames) * stride / sr

    # Apply additional smoothing to f0 using median filter
    from scipy.ndimage import median_filter as scipy_median_filter
    f0_smooth = scipy_median_filter(f0, size=5)
    # Only smooth non-zero values
    f0[f0 > 0] = f0_smooth[f0 > 0]

    # Automatic start/end detection based on peak energy
    # Calculate threshold as a percentage of maximum peak energy
    peak_magnitudes = []
    for frame_idx in range(num_frames):
        if f0[frame_idx] > 0:
            frame_coeffs = coeffs_abs[freq_mask, frame_idx]
            # Find the magnitude at the detected frequency
            detected_freq = f0[frame_idx] * (2.0 if divide_by_2 else 1.0)
            freq_idx = np.argmin(np.abs(fc_search - detected_freq))
            peak_magnitudes.append(frame_coeffs[freq_idx])
        else:
            peak_magnitudes.append(0.0)

    peak_magnitudes = np.array(peak_magnitudes)

    if len(peak_magnitudes[peak_magnitudes > 0]) > 0:
        # Set threshold as percentage of max peak magnitude
        max_peak = np.max(peak_magnitudes[peak_magnitudes > 0])
        threshold = energy_threshold * max_peak

        # Set f0 to 0 where peak is below threshold
        low_energy_mask = peak_magnitudes < threshold
        f0[low_energy_mask] = 0.0
        confidence[low_energy_mask] = 0.0

    # Apply median filtering to smooth out outliers
    confidence_threshold = np.percentile(confidence[confidence > 0], 25) if np.any(confidence > 0) else 0.1
    f0 = median_filter_pitch(f0, confidence, kernel_size=5, conf_threshold=confidence_threshold)

    return time, f0, confidence


def extract_f0_from_dataset(
    audio_dir,
    sr=16000,
    frame_resolution=0.016,
    fmin=10,
    fmax=40,
    use_pitch_shift=False,
    pitch_shift_octaves=2,
    extract_f1=False,
    use_elelet=False,
    elelet_fmin=10,
    elelet_fmax=50,
    elelet_divide_by_2=False,
    elelet_max_jump=1.0,
    elelet_use_global_peak=False,
    elelet_energy_threshold=0.3
):
    """
    Extract f0 from preprocessed 16kHz audio files

    Args:
        audio_dir: Directory with 16kHz audio files
        sr: Sampling rate (must match audio files, default 16000)
        frame_resolution: seconds per frame (for pYIN methods)
        fmin: minimum f0 (for pYIN methods)
        fmax: maximum f0 (for pYIN methods)
        use_pitch_shift: if True, use pitch shifting for better f0 extraction
        pitch_shift_octaves: number of octaves to shift up (default: 2)
        extract_f1: if True, extract F1 (2×F0) then divide by 2 using pYIN
        use_elelet: if True, use Elelet-based peak detection instead of pYIN
        elelet_fmin: minimum frequency to search (default: 15 Hz)
        elelet_fmax: maximum frequency to search (default: 50 Hz)
        elelet_divide_by_2: if True, divide Elelet peak by 2 (default: False)
        elelet_max_jump: max frequency jump between frames in Hz (default: 5.0)
        elelet_use_global_peak: if True, use two-stage approach (default: False)
        elelet_energy_threshold: fraction of max peak for auto start/end (default: 0.3)
    """
    audio_dir = Path(audio_dir)

    # Create f0 directory
    f0_dir = audio_dir / f"f0_{frame_resolution:.3f}"
    f0_dir.mkdir(parents=True, exist_ok=True)

    # Find all WAV files (assuming already preprocessed to 16kHz)
    audio_files = sorted(list(audio_dir.glob("*.wav")))

    if len(audio_files) == 0:
        print(f"ERROR: No WAV files found in {audio_dir}")
        return

    print("="*60)
    print("F1 EXTRACTION")
    print("="*60)
    print(f"Input:  {audio_dir}")
    print(f"Output: {f0_dir}")
    print(f"Files:  {len(audio_files)}")
    print(f"\nSettings:")
    print(f"  Sampling rate: {sr} Hz (assuming audio is already at this rate)")
    if use_elelet:
        # print(f"  F0 method: Elelet peak detection")
        # print(f"    → High-pass filter: {elelet_high_pass:.0f} Hz")
        print(f"    → Finding highest peak in {elelet_fmin:.0f}-{elelet_fmax:.0f} Hz")
        # if elelet_divide_by_2:
        #     print(f"    → Dividing by 2 to get F0 (F1->F0 conversion)")
        # else:
        #     print(f"    → Using peak frequency directly (no division)")
        # if elelet_use_global_peak:
        #     print(f"    → Two-stage: global peak → track with {elelet_max_jump:.1f}Hz max jump")
        # else:
        #     print(f"    → Frame-by-frame with max jump = {elelet_max_jump:.1f} Hz")
        # print(f"    → Stride: {int(sr * frame_resolution)} samples")
    # else:
        # print(f"  Frame resolution: {frame_resolution} s ({1/frame_resolution:.1f} Hz)")
        # print(f"  F0 range: {fmin}-{fmax} Hz")
        # if extract_f1:
        #     print(f"  F0 method: Extract F1 (first overtone) then divide by 2")
        #     print(f"    → Extracting at {fmin*2:.0f}-{fmax*2:.0f} Hz (F1 range), then dividing by 2")
        #     print(f"    → More robust when F1 has more energy than F0")
        # elif use_pitch_shift:
        #     print(f"  F0 method: Pitch shift ({pitch_shift_octaves} octaves up)")
        #     print(f"    → Extracting at {fmin * (2**pitch_shift_octaves):.0f}-{fmax * (2**pitch_shift_octaves):.0f} Hz, then dividing by {2**pitch_shift_octaves}")
        # else:
        #     print(f"  F0 method: Direct pYIN")
    print("="*60 + "\n")

    successful = 0
    failed = []

    #i = 0

    for audio_path in tqdm(audio_files, desc="Extracting f1"):
        try:
            # Load audio (no resampling, assume it's already at target SR)
            y, file_sr = sf.read(str(audio_path))

            # Verify sampling rate
            if file_sr != sr:
                print(f"\n  WARNING: {audio_path.name} has SR={file_sr}, expected {sr}")
                print(f"           Resampling to {sr} Hz...")
                y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

            # Extract f0
            if use_elelet:
                stride = int(sr * frame_resolution)
                time, f0, confidence = extract_f0_elelet(
                    y, sr=sr, stride=stride,
                    fmin=elelet_fmin, fmax=elelet_fmax,
                    divide_by_2=elelet_divide_by_2,
                    max_jump=elelet_max_jump,
                    use_global_peak=elelet_use_global_peak,
                    energy_threshold=elelet_energy_threshold
                )
            else:
                time, f0, confidence = extract_f0_pyin(
                    y, sr=sr, frame_resolution=frame_resolution,
                    fmin=fmin, fmax=fmax,
                    use_pitch_shift=use_pitch_shift,
                    pitch_shift_octaves=pitch_shift_octaves,
                    extract_f1=extract_f1
                )

            # Calculate start/end from first/last non-zero f0
            nonzero_indices = np.where(f0 > 0)[0]
            if len(nonzero_indices) > 0:
                start_time = time[nonzero_indices[0]]
                end_time = time[nonzero_indices[-1]]
            else:
                start_time = 0.0
                end_time = 0.0

            # Create columns with start/end for each row
            start_point_col = np.full(len(time), start_time)
            end_point_col = np.full(len(time), end_time)

            # Save f0 to CSV
            csv_path = f0_dir / f"{audio_path.stem}.f0.csv"
            df = pd.DataFrame({
                'time': time,
                'frequency': f0,
                'start_point': start_point_col,
                'end_point': end_point_col
            })
            df.to_csv(csv_path, index=False)

            successful += 1

        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            failed.append(audio_path.name)
            continue

        #i += 1
        #if i >= 20:
        #    break

    print("\n" + "="*60)
    print("F0 EXTRACTION COMPLETE")
    print("="*60)
    print(f"✓ Successfully processed: {successful}/{len(audio_files)} files")
    if failed:
        print(f"✗ Failed: {len(failed)} files")
        for f in failed:
            print(f"  - {f}")
    print(f"\nOutput: {f0_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Extract f0 from preprocessed 16kHz audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract f0 with default parameters (direct pYIN)
  python extract_f0.py --input data/rumbles

  # Extract F1 (first overtone) then divide by 2 (RECOMMENDED for elephant rumbles)
  # More robust when F1 has more energy than F0
  python extract_f0.py --input data/rumbles --extract_f1

  # Extract f0 using Elelet peak detection (NEW METHOD)
  # Finds highest peak in F1 range (15-50Hz) then divides by 2
  python extract_f0.py --input data/rumbles --use_elelet

  # Extract f0 with Elelet and custom F1 range
  python extract_f0.py --input data/rumbles --use_elelet --elelet_fmin 20 --elelet_fmax 60

  # Extract f0 with pitch shifting
  python extract_f0.py --input data/rumbles --use_pitch_shift

  # Extract f0 with custom parameters
  python extract_f0.py \\
      --input data/rumbles \\
      --fmin 5 \\
      --fmax 100 \\
      --frame_resolution 0.016 \\
      --extract_f1

  # Test different parameters quickly (audio already resampled)
  python extract_f0.py --input data/rumbles --fmin 10 --fmax 80 --extract_f1
  python extract_f0.py --input data/rumbles --use_pitch_shift --pitch_shift_octaves 3
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with 16kHz WAV files')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate (default: 16000)')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution in seconds (default: 0.016)')
    parser.add_argument('--fmin', type=float, default=22.5,
                        help='Minimum f0 frequency (Hz, default: 10)')
    parser.add_argument('--fmax', type=float, default=50,
                        help='Maximum f0 frequency (Hz, default: 40)')
    parser.add_argument('--extract_f1', action='store_true',
                        help='Extract F1 (first overtone at 2×F0) then divide by 2. Recommended when F1 is stronger than F0.')
    parser.add_argument('--use_elelet', type=bool, default=True,
                        help='Use Elelet spectrogram for peak detection instead of pYIN. Finds highest peak in F1 range.')
    parser.add_argument('--elelet_fmin', type=float, default=22.5,
                        help='Minimum F1 frequency for Elelet peak search (Hz, default: 15)')
    parser.add_argument('--elelet_fmax', type=float, default=50,
                        help='Maximum F1 frequency for Elelet peak search (Hz, default: 50)')
    parser.add_argument('--elelet_divide_by_2', action='store_true',
                        help='Divide detected Elelet peak by 2 (for F1->F0 conversion, default: False)')
    parser.add_argument('--elelet_max_jump', type=float, default=1,
                        help='Maximum frequency jump between frames for Elelet (Hz, default: 5.0)')
    parser.add_argument('--elelet_use_global_peak', type=bool, default=True,
                        help='Use two-stage approach: find global peak, then track with 1Hz max jump (default: False)')
    parser.add_argument('--elelet_energy_threshold', type=float, default=0.2,
                        help='Energy threshold for automatic start/end detection (default: 0.3 = 30%% of max peak)')

    args = parser.parse_args()

    extract_f0_from_dataset(
        audio_dir=args.input,
        sr=args.sr,
        frame_resolution=args.frame_resolution,
        fmin=args.fmin,
        fmax=args.fmax,
        use_pitch_shift=args.use_pitch_shift,
        pitch_shift_octaves=args.pitch_shift_octaves,
        extract_f1=args.extract_f1,
        use_elelet=args.use_elelet,
        elelet_fmin=args.elelet_fmin,
        elelet_fmax=args.elelet_fmax,
        elelet_divide_by_2=args.elelet_divide_by_2,
        elelet_max_jump=args.elelet_max_jump,
        elelet_use_global_peak=args.elelet_use_global_peak,
        elelet_energy_threshold=args.elelet_energy_threshold
    )


if __name__ == "__main__":
    main()
