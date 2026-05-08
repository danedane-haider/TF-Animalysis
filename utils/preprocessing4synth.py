"""
Summary preprocessing pipeline:
1) Copy input folder
2) Resample WAVs
3) Extract f0
4) Launch interactive f0 correction
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.resample import resample_dataset
from preprocessing.extract_f0 import extract_f0_from_dataset


def preprocess_summary(
    input_dir,
    copy_input=True,
    overwrite=False,
    output_dir=None,
    sr=16000,
    frame_resolution=0.016,
    fmin=22.5,
    fmax=50.0,
    use_elelet=True,
    elelet_fmin=22.5,
    elelet_fmax=50.0,
    elelet_divide_by_2=False,
    elelet_max_jump=1.0,
    elelet_use_global_peak=True,
    elelet_energy_threshold=0.2,
    extract_f1=False,
    use_pitch_shift=False,
    pitch_shift_octaves=2,
    run_annotate=True,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if output_dir is None:
        work_dir = input_dir.parent / f"{input_dir.name}_synth"
    else:
        work_dir = output_dir / f"{input_dir.name}_synth"

    if work_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Work directory already exists: {work_dir}")
        shutil.rmtree(work_dir)

    print("=" * 60)
    print("PREPROCESS SUMMARY")
    print("=" * 60)
    print(f"Input:      {input_dir}")
    print(f"Work dir:   {work_dir}")
    print("=" * 60 + "\n")

    if copy_input:
        print("Step 1/3: Copying input folder...")
        shutil.copytree(input_dir, work_dir)
    else:
        print("Step 1/3: Skipping copy (using input folder directly)...")

    print("Step 2/3: Resampling audio...")
    resample_dataset(
        input_dir=work_dir if copy_input else input_dir,
        output_dir=work_dir,
        sr=sr,
    )

    print("Step 3/3: Extracting f0...")
    extract_f0_from_dataset(
        audio_dir=work_dir,
        sr=sr,
        frame_resolution=frame_resolution,
        fmin=fmin,
        fmax=fmax,
        use_pitch_shift=use_pitch_shift,
        pitch_shift_octaves=pitch_shift_octaves,
        extract_f1=extract_f1,
        use_elelet=use_elelet,
        elelet_fmin=elelet_fmin,
        elelet_fmax=elelet_fmax,
        elelet_divide_by_2=elelet_divide_by_2,
        elelet_max_jump=elelet_max_jump,
        elelet_use_global_peak=elelet_use_global_peak,
        elelet_energy_threshold=elelet_energy_threshold,
    )

    if run_annotate:
        print("\nLaunching interactive f0 correction (annotate_f0.py)...")
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "annotate_f0.py"),
            "--input",
            str(work_dir),
            "--sr",
            str(sr),
            "--frame_resolution",
            str(frame_resolution),
        ]
        subprocess.run(cmd, check=False)
    else:
        print("\nSkipping annotate_f0.py (run manually if needed).")
        print(
            f"Command: {sys.executable} preprocessing/annotate_f0.py "
            f"--input {resampled_dir} --sr {sr} --frame_resolution {frame_resolution}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summary preprocessing: copy, resample, extract f0, annotate f0"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input folder with raw audio files")
    parser.add_argument("--output", type=str, default=None,
                        help="Parent output directory for the *_synth folder")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Target sampling rate (default: 16000)")
    parser.add_argument("--no_copy", dest="copy_input", action="store_false",
                        help="Do not copy input folder; operate directly on input")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite work_dir if it already exists")
    parser.add_argument("--frame_resolution", type=float, default=0.016,
                        help="Frame resolution in seconds (default: 0.016)")
    parser.add_argument("--fmin", type=float, default=22.5,
                        help="Minimum f0 frequency (Hz, default: 22.5)")
    parser.add_argument("--fmax", type=float, default=50.0,
                        help="Maximum f0 frequency (Hz, default: 50)")
    parser.add_argument("--use_elelet", action="store_true", default=True,
                        help="Use Elelet peak detection (default: True)")
    parser.add_argument("--no_elelet", dest="use_elelet", action="store_false",
                        help="Disable Elelet peak detection")
    parser.add_argument("--elelet_fmin", type=float, default=22.5,
                        help="Minimum F1 frequency for Elelet (Hz, default: 22.5)")
    parser.add_argument("--elelet_fmax", type=float, default=50.0,
                        help="Maximum F1 frequency for Elelet (Hz, default: 50)")
    parser.add_argument("--elelet_divide_by_2", action="store_true",
                        help="Divide Elelet peak by 2 (F1->F0)")
    parser.add_argument("--elelet_max_jump", type=float, default=1.0,
                        help="Max frequency jump between frames (Hz, default: 1.0)")
    parser.add_argument("--elelet_use_global_peak", action="store_true", default=True,
                        help="Use global peak tracking (default: True)")
    parser.add_argument("--no_elelet_global_peak", dest="elelet_use_global_peak",
                        action="store_false",
                        help="Disable global peak tracking")
    parser.add_argument("--elelet_energy_threshold", type=float, default=0.2,
                        help="Energy threshold for auto start/end (default: 0.2)")
    parser.add_argument("--extract_f1", action="store_true",
                        help="Extract F1 then divide by 2 (pYIN mode)")
    parser.add_argument("--use_pitch_shift", action="store_true",
                        help="Use pitch shifting (pYIN mode)")
    parser.add_argument("--pitch_shift_octaves", type=int, default=2,
                        help="Pitch shift octaves (default: 2)")
    parser.add_argument("--no_annotate", dest="run_annotate", action="store_false",
                        help="Skip launching annotate_f0.py")

    args = parser.parse_args()

    preprocess_summary(
        input_dir=args.input,
        copy_input=args.copy_input,
        overwrite=args.overwrite,
        output_dir=args.output,
        sr=args.sr,
        frame_resolution=args.frame_resolution,
        fmin=args.fmin,
        fmax=args.fmax,
        use_elelet=args.use_elelet,
        elelet_fmin=args.elelet_fmin,
        elelet_fmax=args.elelet_fmax,
        elelet_divide_by_2=args.elelet_divide_by_2,
        elelet_max_jump=args.elelet_max_jump,
        elelet_use_global_peak=args.elelet_use_global_peak,
        elelet_energy_threshold=args.elelet_energy_threshold,
        extract_f1=args.extract_f1,
        use_pitch_shift=args.use_pitch_shift,
        pitch_shift_octaves=args.pitch_shift_octaves,
        run_annotate=args.run_annotate,
    )


if __name__ == "__main__":
    main()
