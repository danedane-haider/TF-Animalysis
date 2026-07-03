"""Extract F1 contours with Elelet spectral peak tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f0_extraction.extract_f0 import extract_f0_from_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract F1 contours by tracking spectral peaks in Elelet space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Directory with WAV files.")
    parser.add_argument("--output_dir_name", type=str, default=None, help="Output directory under input dir.")
    parser.add_argument(
        "--use_precomputed_representations",
        type=str,
        default=None,
        help="Path to a precomputed Elelet folder, or a parent containing elelet_*.",
    )
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="elelet",
        help="Output label for comparing algorithms, e.g. elelet_peak_v2.",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--frame_resolution", type=float, default=0.016, help="Frame resolution in seconds.")
    parser.add_argument("--f1_min", type=float, default=22.5, help="Minimum F1 frequency to track.")
    parser.add_argument("--f1_max", type=float, default=50.0, help="Maximum F1 frequency to track.")
    parser.add_argument("--max_jump", type=float, default=1.0, help="Maximum frame-to-frame F1 jump in Hz.")
    parser.add_argument("--no_max_jump", dest="max_jump", action="store_const", const=None, help="Disable continuity limit.")
    parser.add_argument("--global_peak", action=argparse.BooleanOptionalAction, default=True, help="Track outward from strongest frame.")
    parser.add_argument("--energy_threshold", type=float, default=0.2, help="Fraction of max peak magnitude used to zero weak frames.")
    parser.add_argument("--divide_by_2", action="store_true", help="Store F1/2 instead of F1.")
    parser.add_argument("--median_kernel", type=int, default=5, help="Median smoothing kernel.")
    parser.add_argument("--num_channels", type=int, default=1024, help="Elelet channels.")
    parser.add_argument("--kernel_size", type=int, default=24000, help="Elelet kernel size.")
    parser.add_argument("--transform_fmin", type=float, default=5.0, help="Elelet transform minimum frequency.")
    parser.add_argument("--transform_fmax", type=float, default=100.0, help="Elelet transform maximum frequency.")
    parser.add_argument("--supp_mult", type=float, default=0.2, help="Elelet support multiplier.")
    parser.add_argument("--scale", type=str, default="elelog", help="Elelet frequency scale.")
    args = parser.parse_args()

    extract_f0_from_dataset(
        audio_dir=args.input,
        sr=args.sr,
        frame_resolution=args.frame_resolution,
        pipeline="elelet",
        algorithm_name=args.algorithm_name,
        output_dir_name=args.output_dir_name,
        f1_min=args.f1_min,
        f1_max=args.f1_max,
        max_jump=args.max_jump,
        use_global_peak=args.global_peak,
        energy_threshold=args.energy_threshold,
        divide_by_2=args.divide_by_2,
        median_kernel=args.median_kernel,
        elelet_num_channels=args.num_channels,
        elelet_kernel_size=args.kernel_size,
        elelet_fmin=args.transform_fmin,
        elelet_fmax=args.transform_fmax,
        elelet_supp_mult=args.supp_mult,
        elelet_scale=args.scale,
        use_precomputed_representations=args.use_precomputed_representations,
    )


if __name__ == "__main__":
    main()
