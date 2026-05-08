"""
Resample audio files to target sampling rate
Converts any audio format to 16kHz WAV
"""

import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse


def resample_dataset(input_dir, output_dir, sr=16000):
    """
    Resample all audio files to target sampling rate

    Args:
        input_dir: Directory with raw audio (any format/SR)
        output_dir: Directory for resampled audio
        sr: Target sampling rate (default: 16000)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_extensions = ['*.wav', '*.WAV', '*.mp3', '*.flac', '*.ogg', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(input_dir.glob(ext)))

    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {input_dir}")
        return

    print("="*60)
    print("AUDIO RESAMPLING")
    print("="*60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(audio_files)}")
    print(f"Target sampling rate: {sr} Hz")
    print("="*60 + "\n")

    successful = 0
    failed = []

    for audio_path in tqdm(audio_files, desc="Resampling"):
        try:
            # Load audio (librosa auto-resamples if needed)
            y, orig_sr = librosa.load(str(audio_path), sr=sr)
            if y.shape[-1] == 2:
                y = librosa.to_mono(y)

            if len(y) < sr:
                y = librosa.util.fix_length(y, size=sr)

            # Save resampled audio as WAV
            audio_out = output_dir / audio_path.name
            audio_out = audio_out.with_suffix('.wav')
            sf.write(audio_out, y, sr)

            successful += 1

        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            failed.append(audio_path.name)
            continue

    print("\n" + "="*60)
    print("RESAMPLING COMPLETE")
    print("="*60)
    print(f"✓ Successfully processed: {successful}/{len(audio_files)} files")
    if failed:
        print(f"✗ Failed: {len(failed)} files")
        for f in failed:
            print(f"  - {f}")
    print(f"\nOutput: {output_dir}/")
    print(f"\nNext step:")
    print(f"  python extract_f0.py --input {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Resample audio files to target sampling rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resample to 16kHz (default)
  python resample_audio.py --input raw_data/rumbles --output data/rumbles

  # Resample to different rate
  python resample_audio.py --input raw_data/rumbles --output data/rumbles --sr 22050
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with raw audio (any format/SR)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for resampled audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Target sampling rate (Hz, default: 16000)')

    args = parser.parse_args()

    resample_dataset(
        input_dir=args.input,
        output_dir=args.output,
        sr=args.sr
    )


if __name__ == "__main__":
    main()
