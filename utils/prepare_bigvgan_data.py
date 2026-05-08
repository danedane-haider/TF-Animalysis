"""
Prepare training data for BigVGAN vocoder.

This script:
1. Loads audio files
2. Computes EleSpectrograms (elephant-specific STFT-based spectrograms)
3. Saves spectrograms in BigVGAN-compatible format (.npy or .pt)
4. Creates file lists for training

BigVGAN will learn to reconstruct audio from EleSpectrograms.
"""

import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.transforms import EleSpectrogram


def compute_elespectrogram(audio, sr=16000,
                           num_channels=128,
                           fmax=1000,
                           hop_length=256,
                           n_fft=4096,
                           fmin=0):
    """
    Compute EleSpectrogram for elephant rumbles.

    Args:
        audio: Audio signal (1D numpy array)
        sr: Sample rate
        num_channels: Number of EleSpectrogram filterbank channels
        fmax: Maximum frequency (Hz)
        hop_length: Hop length in samples (STFT stride)
        n_fft: FFT size
        fmin: Minimum frequency (Hz)

    Returns:
        spec: EleSpectrogram in dB scale (num_channels, num_frames)
    """
    # Convert to torch tensor
    audio_torch = torch.from_numpy(audio).float().unsqueeze(0)

    # Initialize EleSpectrogram transform
    elespec = EleSpectrogram(
        fs=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        num_channels=num_channels,
        scale='elelog',  # Elephant hearing scale
        power=2.0,
    )

    # Compute spectrogram
    with torch.no_grad():
        spec = elespec(audio_torch)  # (1, num_channels, num_frames)

    spec = spec.squeeze(0).numpy()  # (num_channels, num_frames)

    # Convert to dB scale
    spec_db = 10 * np.log10(spec + 1e-8)  # Power -> dB (factor of 10, not 20)

    return spec_db


def process_dataset(audio_dir, output_dir,
                    sr=16000,
                    num_channels=128,
                    fmax=650,
                    hop_length=256,
                    n_fft=2048,
                    fmin=10,
                    format='npy'):
    """
    Process entire dataset and save EleSpectrograms.

    Args:
        audio_dir: Directory containing .wav files
        output_dir: Output directory for spectrograms
        format: 'npy' or 'pt' (PyTorch tensor)
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    # Create output directories
    spec_dir = output_dir / 'elespec'
    spec_dir.mkdir(exist_ok=True, parents=True)

    # Find all audio files
    audio_files = sorted(list(audio_dir.glob('*.wav')))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")

    print("="*60)
    print("PREPARING BIGVGAN TRAINING DATA")
    print("="*60)
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(audio_files)}")
    print(f"EleSpectrogram config:")
    print(f"  Channels: {num_channels}")
    print(f"  Fmin: {fmin} Hz")
    print(f"  Fmax: {fmax} Hz")
    print(f"  Hop length: {hop_length} samples ({hop_length/sr*1000:.1f} ms)")
    print(f"  N_FFT: {n_fft}")
    print()

    # Process files
    file_list = []

    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            # Load audio
            audio, file_sr = sf.read(audio_path)

            # Resample if needed
            if file_sr != sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

            # Compute EleSpectrogram
            spec = compute_elespectrogram(
                audio, sr=sr,
                num_channels=num_channels,
                fmax=fmax,
                hop_length=hop_length,
                n_fft=n_fft,
                fmin=fmin
            )

            # Save spectrogram
            spec_filename = f"{audio_path.stem}.{format}"
            spec_path = spec_dir / spec_filename

            if format == 'npy':
                np.save(spec_path, spec)
            elif format == 'pt':
                torch.save(torch.from_numpy(spec), spec_path)
            else:
                raise ValueError(f"Unknown format: {format}")

            # Add to file list (relative paths)
            file_list.append({
                'audio': str(audio_path.relative_to(audio_dir.parent)),
                'spec': str(spec_path.relative_to(output_dir))
            })

        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save file list
    file_list_path = output_dir / 'train_files.txt'
    with open(file_list_path, 'w') as f:
        for item in file_list:
            f.write(f"{item['audio']}|{item['spec']}\n")

    print()
    print("="*60)
    print("✓ Data preparation complete!")
    print("="*60)
    print(f"Processed: {len(file_list)} files")
    print(f"Spectrograms saved to: {spec_dir}")
    print(f"File list saved to: {file_list_path}")
    print()
    print("Next steps:")
    print(f"  1. Copy/symlink audio files to BigVGAN training directory")
    print(f"  2. Configure BigVGAN to use Elelet spectrograms")
    print(f"  3. Update BigVGAN config with paths to train_files.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare EleSpectrograms for BigVGAN training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare vowels dataset
  python preprocessing/prepare_bigvgan_data.py \\
    --audio_dir data/ntvow_low_train \\
    --output_dir data/bigvgan/vowels_pretrain

  # Prepare rumbles dataset
  python preprocessing/prepare_bigvgan_data.py \\
    --audio_dir data/rumbles \\
    --output_dir data/bigvgan/rumbles_finetune

  # Prepare DDSP synthesized outputs (after DDSP training)
  python preprocessing/prepare_bigvgan_data.py \\
    --audio_dir outputs/ddsp_synthesized \\
    --output_dir data/bigvgan/ddsp_synthesized

What it does:
  1. Loads audio files from audio_dir
  2. Computes EleSpectrograms (STFT-based with elephant hearing scale)
  3. Saves spectrograms in BigVGAN-compatible format
  4. Creates train_files.txt with audio|spec pairs

The EleSpectrograms use elephant hearing scale (10-650 Hz).
        """
    )

    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for spectrograms')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate (default: 16000)')
    parser.add_argument('--num_channels', type=int, default=128,
                        help='Number of EleSpectrogram channels (default: 128)')
    parser.add_argument('--fmin', type=float, default=10,
                        help='Minimum frequency in Hz (default: 10)')
    parser.add_argument('--fmax', type=float, default=650,
                        help='Maximum frequency in Hz (default: 650)')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='STFT hop length in samples (default: 256)')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT size (default: 2048)')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'pt'],
                        help='Output format: npy or pt (default: npy)')

    args = parser.parse_args()

    process_dataset(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        sr=args.sr,
        num_channels=args.num_channels,
        fmax=args.fmax,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        fmin=args.fmin,
        format=args.format
    )


if __name__ == "__main__":
    main()
