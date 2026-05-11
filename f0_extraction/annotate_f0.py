"""
Interactive f0 correction tool with STFT and Elelet spectrogram options
Click on spectrogram to add correction points, interpolates between clicks
"""

import argparse
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Button, CheckButtons
from scipy.interpolate import interp1d
import soundfile as sf
import sounddevice as sd
import sys

# Import Elelet transform and utilities
sys.path.append(str(Path(__file__).parent.parent))
from f0_extraction.pipeline import (
    corrected_dir as default_corrected_dir,
    extracted_dir,
    normalize_algorithm,
    representation_dir,
    resolve_existing_dir,
)


class F0Corrector:
    def __init__(self, audio_dir, sr=16000, frame_resolution=0.016,
                 n_fft=8192, hop_length=256, fmin=0, fmax=100, start_idx=0,
                 use_precomputed=True, algorithm_name=None,
                 f0_dir=None, corrected_dir=None,
                 stft_dir=None, elelet_dir=None, initial_spec_mode=None,
                 representation_fmax=750, enable_elelet_view=True, max_harmonic=10):
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.frame_resolution = frame_resolution
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.use_precomputed = use_precomputed
        self.algorithm_name = normalize_algorithm(algorithm_name) if algorithm_name else None
        self.max_harmonic = max(2, int(max_harmonic))

        # Spectrogram mode: 'stft' or 'elelet'
        self.spec_mode = (initial_spec_mode or "stft").lower().strip()
        if self.spec_mode not in ("stft", "elelet"):
            raise ValueError("initial_spec_mode must be 'stft' or 'elelet'")
        self.enable_elelet_view = enable_elelet_view or self.spec_mode == "elelet"

        # Pre-computed representation directories
        self.stft_dir = self.audio_dir / (stft_dir or representation_dir("stft", representation_fmax))
        self.elelet_dir = self.audio_dir / (elelet_dir or representation_dir("elelet", representation_fmax))

        # Check if pre-computed directories exist
        if self.use_precomputed:
            active_dir = self.stft_dir if self.spec_mode == "stft" else self.elelet_dir
            if not active_dir.exists():
                print(f"Warning: Pre-computed directories not found!")
                print(f"  STFT: {self.stft_dir.exists()} - {self.stft_dir}")
                print(f"  Elelet: {self.elelet_dir.exists()} - {self.elelet_dir}")
                print(f"  Falling back to on-the-fly computation")
                print(f"  Run the matching precompute_representations_* script first to pre-compute")
                self.use_precomputed = False

        self.elelet_transform = None
        if self.enable_elelet_view:
            self._init_elelet_transform()

        # Find f0 directory
        self.f0_dir = resolve_existing_dir(
            self.audio_dir,
            f0_dir,
            extracted_dir(self.algorithm_name or "elelet"),
            legacy_name=f"f0_{frame_resolution:.3f}",
        )
        if not self.f0_dir.exists():
            raise ValueError(f"F0 directory not found: {self.f0_dir}")

        # Corrected contours always converge into one human-reviewed folder.
        self.corrected_dir = self.audio_dir / (
            corrected_dir or default_corrected_dir()
        )
        self.corrected_dir.mkdir(exist_ok=True)
        self.contour_dirs = self._find_contour_dirs()

        # Find all audio files
        self.audio_files = sorted(list(self.audio_dir.glob("*.wav")))
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {self.audio_dir}")

        # Validate start index
        if start_idx < 0 or start_idx >= len(self.audio_files):
            print(f"Warning: start index {start_idx} out of range [0, {len(self.audio_files)-1}], using 0")
            start_idx = 0

        # Current state
        self.current_idx = 0
        self.correction_points = []  # List of (time, freq) tuples
        self.start_point = None  # (time, freq) for start of region
        self.end_point = None  # (time, freq) for end of region
        self.original_f0 = None
        self.corrected_f0 = None
        self.show_f0_contours = True
        self.show_upper_harmonics = False
        self.contour_visibility = {
            "original": True,
            "corrected": True,
        }
        self.contour_order = []
        self.contour_check_ax = None
        self.contour_check = None
        self.contour_check_keys = []
        self.contour_check_labels = []

        # Cached spectrograms (computed on-demand for speed)
        self.stft_cached = False
        self.elelet_cached = False

        # Setup figure (single plot for spectrogram only)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8))
        plt.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.12)  # Make room for buttons and contour toggles
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Add buttons
        self.setup_buttons()

        # Load file at start index
        self.load_file(start_idx)

    def _find_contour_dirs(self):
        """Find all f0-like directories that contain contour CSVs."""
        contour_dirs = []
        for path in sorted(self.audio_dir.iterdir()):
            if not path.is_dir() or not path.name.startswith("f0"):
                continue
            if any(path.glob("*.f0.csv")):
                contour_dirs.append(path)
        return contour_dirs

    def _load_contour_versions(self, audio_stem):
        """Load every available contour version for the current audio file."""
        versions = {}
        for contour_dir in self.contour_dirs:
            if contour_dir.resolve() in {self.f0_dir.resolve(), self.corrected_dir.resolve()}:
                continue
            csv_path = contour_dir / f"{audio_stem}.f0.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"Warning: could not load {csv_path}: {exc}")
                continue
            if {"time", "frequency"}.issubset(df.columns):
                versions[contour_dir.name] = (
                    df["time"].to_numpy(dtype=float),
                    df["frequency"].to_numpy(dtype=float),
                )
        return versions

    def _display_contour_name(self, key):
        if key == "original":
            return "original"
        if key == "corrected":
            return "corrected"
        if key.startswith("f0_"):
            return key[3:]
        return key

    def _sync_contour_visibility(self):
        """Keep per-contour visibility controls in sync with available files."""
        self.contour_order = ["original", "corrected", *self.other_contours.keys()]
        for key in self.contour_order:
            self.contour_visibility.setdefault(key, True)

    def _contour_title_summary(self):
        if not self.show_f0_contours:
            return "OFF"
        visible = [
            self._display_contour_name(key)
            for key in self.contour_order
            if self.contour_visibility.get(key, True)
        ]
        if not visible:
            return "NONE"
        if len(visible) == len(self.contour_order):
            return "ALL"
        if len(visible) <= 3:
            return ", ".join(visible)
        return ", ".join(visible[:3]) + ", ..."

    def _refresh_contour_checkboxes(self):
        """Build the small GUI panel for toggling individual contour versions."""
        if self.contour_check_ax is not None:
            self.contour_check_ax.remove()

        if not self.contour_order:
            self.contour_check_ax = None
            self.contour_check = None
            self.contour_check_keys = []
            self.contour_check_labels = []
            return

        height = min(0.038 * len(self.contour_order) + 0.045, 0.30)
        bottom = 0.92 - height
        self.contour_check_ax = plt.axes([0.895, bottom, 0.085, height])
        self.contour_check_ax.set_title("Ctrs", fontsize=8)
        self.contour_check_keys = list(self.contour_order)
        self.contour_check_labels = [
            f"{idx + 1}. {self._display_contour_name(key)}"
            for idx, key in enumerate(self.contour_check_keys)
        ]
        active = [
            self.contour_visibility.get(key, True)
            for key in self.contour_check_keys
        ]
        self.contour_check = CheckButtons(
            self.contour_check_ax,
            self.contour_check_labels,
            active,
        )
        for label in self.contour_check.labels:
            label.set_fontsize(7)
        self.contour_check.on_clicked(self._on_contour_check)

    def _on_contour_check(self, label):
        if label not in self.contour_check_labels:
            return
        idx = self.contour_check_labels.index(label)
        key = self.contour_check_keys[idx]
        self.contour_visibility[key] = self.contour_check.get_status()[idx]
        if self.contour_visibility[key]:
            self.show_f0_contours = True
        status = "ON" if self.contour_visibility[key] else "OFF"
        print(f"→ {self._display_contour_name(key)} contour: {status}")
        self.update_plot()

    def _init_elelet_transform(self):
        """Initialize the Elelet transform only for workflows that need it."""
        if self.elelet_transform is not None:
            return

        import torch
        from tf_transforms.transforms import Elelet

        self.elelet_transform = Elelet(
            kernel_size=16000+8000,
            num_channels=1024,
            stride=256,
            fmin=5,
            fmax=500,
            fs=16000,
            supp_mult=0.2,
            scale='elelog',
        )

        if isinstance(self.elelet_transform.fc, torch.Tensor):
            self.elelet_transform.fc = self.elelet_transform.fc.numpy()

    def setup_buttons(self):
        """Setup navigation and action buttons"""
        # Previous button
        ax_prev = plt.axes([0.05, 0.01, 0.07, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous (←)')
        self.btn_prev.on_clicked(lambda e: self.load_file(self.current_idx - 1))

        # Next button
        ax_next = plt.axes([0.13, 0.01, 0.07, 0.04])
        self.btn_next = Button(ax_next, 'Next (→)')
        self.btn_next.on_clicked(lambda e: self.load_file(self.current_idx + 1))

        # Play button
        ax_play = plt.axes([0.21, 0.01, 0.07, 0.04])
        self.btn_play = Button(ax_play, 'Play (␣)')
        self.btn_play.on_clicked(lambda e: self.play_audio())

        # Toggle spectrogram button
        ax_toggle = plt.axes([0.29, 0.01, 0.13, 0.04])
        toggle_label = 'Switch Repr. (R)' if self.enable_elelet_view else 'STFT View'
        self.btn_toggle = Button(ax_toggle, toggle_label)
        self.btn_toggle.on_clicked(lambda e: self.toggle_spectrogram())

        # Mark Start button
        ax_start = plt.axes([0.43, 0.01, 0.09, 0.04])
        self.btn_start = Button(ax_start, 'Start Line (W)')
        self.btn_start.on_clicked(lambda e: self.set_marking_mode('start'))

        # Mark End button
        ax_end = plt.axes([0.53, 0.01, 0.09, 0.04])
        self.btn_end = Button(ax_end, 'End Line (E)')
        self.btn_end.on_clicked(lambda e: self.set_marking_mode('end'))

        # Clear corrections button
        ax_clear = plt.axes([0.63, 0.01, 0.09, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear (C)')
        self.btn_clear.on_clicked(lambda e: self.clear_corrections())

        # Display controls
        ax_contours = plt.axes([0.73, 0.01, 0.09, 0.04])
        self.btn_contours = Button(ax_contours, 'All Ctrs (F)')
        self.btn_contours.on_clicked(lambda e: self.toggle_f0_contours())

        ax_harmonics = plt.axes([0.83, 0.01, 0.11, 0.04])
        self.btn_harmonics = Button(ax_harmonics, 'Harmonics (H)')
        self.btn_harmonics.on_clicked(lambda e: self.toggle_upper_harmonics())

        # Quit button
        ax_quit = plt.axes([0.95, 0.01, 0.04, 0.04])
        self.btn_quit = Button(ax_quit, 'Q')
        self.btn_quit.on_clicked(lambda e: self.quit_tool())

        # Marking mode
        self.marking_mode = None  # None, 'start', or 'end'

    def toggle_spectrogram(self):
        """Toggle between STFT and Elelet spectrogram representations"""
        if self.spec_mode == 'stft':
            if not self.enable_elelet_view:
                print("→ Elelet view is disabled for this STFT workflow")
                return
            self._init_elelet_transform()
            self.spec_mode = 'elelet'
            print("→ Switched to ELELET spectrogram")
        else:
            self.spec_mode = 'stft'
            print("→ Switched to STFT spectrogram")
        self.update_plot()

    def toggle_f0_contours(self):
        """Toggle all F0 contour overlays."""
        self.show_f0_contours = not self.show_f0_contours
        status = "ON" if self.show_f0_contours else "OFF"
        print(f"→ F0 contours: {status}")
        self.update_plot()

    def toggle_single_contour(self, slot):
        """Toggle one contour by its displayed number."""
        idx = int(slot) - 1
        if idx < 0 or idx >= len(self.contour_order):
            print(f"→ No contour assigned to {slot}")
            return

        key = self.contour_order[idx]
        self.contour_visibility[key] = not self.contour_visibility.get(key, True)
        if self.contour_visibility[key]:
            self.show_f0_contours = True
        status = "ON" if self.contour_visibility[key] else "OFF"
        print(f"→ {self._display_contour_name(key)} contour: {status}")
        self._refresh_contour_checkboxes()
        self.update_plot()

    def toggle_upper_harmonics(self):
        """Toggle harmonic overlays derived from the contour's implied F0."""
        self.show_upper_harmonics = not self.show_upper_harmonics
        status = "ON" if self.show_upper_harmonics else "OFF"
        print(f"→ Upper harmonics: {status}")
        self.update_plot()

    def play_audio(self):
        """Play the current audio sample"""
        print(f"→ Playing audio: {self.audio_files[self.current_idx].name}")
        sd.play(self.audio, self.sr)

    def set_marking_mode(self, mode):
        """Set the marking mode for next click"""
        self.marking_mode = mode
        if mode == 'start':
            print("→ Click on spectrogram to mark START time (green line)")
        elif mode == 'end':
            print("→ Click on spectrogram to mark END time (blue line)")
        else:
            print("Normal correction mode")

    def load_file(self, idx):
        """Load audio file and f0 data"""
        # Auto-save current file before loading new one
        if hasattr(self, 'audio'):  # Skip on first load
            self.save_corrections(quiet=True)

        # Wrap around
        idx = idx % len(self.audio_files)
        self.current_idx = idx

        audio_path = self.audio_files[idx]
        f0_path = self.f0_dir / f"{audio_path.stem}.f0.csv"

        if not f0_path.exists():
            print(f"Warning: F0 file not found for {audio_path.name}")
            return

        # Load audio
        self.audio, _ = librosa.load(audio_path, sr=self.sr)

        # Load f0
        df = pd.read_csv(f0_path)
        self.f0_time = df['time'].values
        self.f0_freq = df['frequency'].values
        # Confidence column is optional (old format), ignore if not present
        if 'confidence' in df.columns:
            self.f0_conf = df['confidence'].values
        else:
            self.f0_conf = np.ones_like(self.f0_freq)  # Default to 1.0

        # Check if corrected version exists
        corrected_path = self.corrected_dir / f"{audio_path.stem}.f0.csv"
        if corrected_path.exists():
            df_corrected = pd.read_csv(corrected_path)
            self.original_f0 = self.f0_freq.copy()
            self.corrected_f0 = df_corrected['frequency'].values

            # Correction points are not stored anymore - reset
            self.correction_points = []

            # Load start/end points if stored
            if 'start_point' in df_corrected.columns:
                start_mask = df_corrected['start_point'] == 1
                if start_mask.any():
                    pt_idx = np.where(start_mask)[0][0]
                    self.start_point = (df_corrected.loc[pt_idx, 'time'],
                                       df_corrected.loc[pt_idx, 'frequency'])
                else:
                    self.start_point = None
            else:
                self.start_point = None
            if 'end_point' in df_corrected.columns:
                end_mask = df_corrected['end_point'] == 1
                if end_mask.any():
                    pt_idx = np.where(end_mask)[0][0]
                    self.end_point = (df_corrected.loc[pt_idx, 'time'],
                                     df_corrected.loc[pt_idx, 'frequency'])
                else:
                    self.end_point = None
            else:
                self.end_point = None

        else:
            self.original_f0 = self.f0_freq.copy()
            self.corrected_f0 = self.f0_freq.copy()
            self.correction_points = []
            self.start_point = None
            self.end_point = None

        self.contour_dirs = self._find_contour_dirs()
        self.other_contours = self._load_contour_versions(audio_path.stem)
        self._sync_contour_visibility()
        self._refresh_contour_checkboxes()

        # Mark spectrograms as not cached (will compute on-demand)
        self.stft_cached = False
        self.elelet_cached = False

        # Update display (will compute needed spectrogram)
        self.update_plot()

    def compute_stft(self):
        """Compute or load STFT spectrogram if not cached"""
        if not self.stft_cached:
            # Try loading pre-computed data first
            if self.use_precomputed:
                audio_path = self.audio_files[self.current_idx]
                precomputed_path = self.stft_dir / f"{audio_path.stem}.npz"
                if precomputed_path.exists():
                    data = np.load(precomputed_path)
                    precomputed_fmax = data.get('fmax', 750)  # Default to 750 if not stored

                    # Only recompute if requested fmax > precomputed fmax
                    if self.fmax > precomputed_fmax:
                        print(f"→ Requested fmax={self.fmax}Hz > precomputed fmax={precomputed_fmax}Hz")
                        print(f"  Recomputing STFT with fmax={self.fmax}Hz...")
                        # Fall through to recomputation
                    else:
                        # Use precomputed data (will be sliced in update_plot)
                        self.S_db = data['S_db']
                        self.spec_times = data['times']
                        self.spec_freqs = data['freqs']
                        self.stft_cached = True
                        return

            # Fall back to on-the-fly computation
            D = librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
            self.S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            self.spec_times = librosa.times_like(self.S_db, sr=self.sr, hop_length=self.hop_length)
            self.spec_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            self.stft_cached = True

    def compute_elelet(self):
        """Compute or load Elelet spectrogram if not cached"""
        if not self.elelet_cached:
            self._init_elelet_transform()
            # Try loading pre-computed data first
            if self.use_precomputed:
                audio_path = self.audio_files[self.current_idx]
                precomputed_path = self.elelet_dir / f"{audio_path.stem}.npz"
                if precomputed_path.exists():
                    data = np.load(precomputed_path)
                    precomputed_fmax = data.get('fmax', 750)  # Default to 750 if not stored

                    # Only recompute if requested fmax > precomputed fmax
                    if self.fmax > precomputed_fmax:
                        print(f"→ Requested fmax={self.fmax}Hz > precomputed fmax={precomputed_fmax}Hz")
                        print(f"  Recomputing Elelet with fmax={self.fmax}Hz...")
                        # Fall through to recomputation
                    else:
                        # Use precomputed data (will be sliced in update_plot)
                        self.elelet_coeffs = data['coeffs']
                        self.elelet_coeffs_abs = data['coeffs_abs']
                        self.elelet_times = data['times']
                        # Update transform's fc if available in precomputed data
                        if 'fc' in data:
                            self.elelet_transform.fc = data['fc']
                        self.elelet_cached = True
                        return

            # Fall back to on-the-fly computation
            self.elelet_coeffs = self.elelet_transform(self.audio)
            self.elelet_coeffs_abs = np.abs(self.elelet_coeffs)
            num_frames = self.elelet_coeffs.shape[1]
            num_channels = self.elelet_coeffs.shape[0]
            self.elelet_times = np.arange(num_frames) * self.elelet_transform.stride / self.sr
            self.elelet_cached = True

    def _contour_style(self, idx, name):
        is_refined = "refined" in name
        color = "deeppink" if is_refined else plt.cm.tab10(idx % 10)
        linestyle = "-" if is_refined else "-"
        linewidth = 2.0 if is_refined else 1.1
        alpha = 0.95 if is_refined else 0.6
        return color, linestyle, linewidth, alpha

    def _contour_series(self, require_master_toggle=True):
        if self.original_f0 is not None:
            if self.contour_visibility.get("original", True):
                if self.show_f0_contours or not require_master_toggle:
                    yield (
                        "original",
                        "Original",
                        self.f0_time,
                        self.original_f0,
                        "gray",
                        "-",
                        1.5,
                        0.55,
                    )

        if self.corrected_f0 is not None:
            if self.contour_visibility.get("corrected", True):
                if self.show_f0_contours or not require_master_toggle:
                    yield (
                        "corrected",
                        "Corrected",
                        self.f0_time,
                        self.corrected_f0,
                        "c",
                        "-",
                        2.0,
                        0.85,
                    )

        for idx, (name, (times, freqs)) in enumerate(self.other_contours.items()):
            if not self.contour_visibility.get(name, True):
                continue
            if not self.show_f0_contours and require_master_toggle:
                continue
            color, linestyle, linewidth, alpha = self._contour_style(idx, name)
            yield (
                name,
                self._display_contour_name(name),
                times,
                freqs,
                color,
                linestyle,
                linewidth,
                alpha,
            )

    def _plot_h1_and_f0_stft(self):
        """Plot each visible CSV H1 contour plus its derived F0 = H1 / 2."""
        for _, label, times, h1_freqs, color, linestyle, linewidth, alpha in self._contour_series():
            valid_h1 = h1_freqs > 0
            if not np.any(valid_h1):
                continue

            self.ax.plot(
                times[valid_h1],
                h1_freqs[valid_h1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=f"{label} H1",
            )

            f0_freqs = h1_freqs / 2.0
            valid_f0 = f0_freqs > 0
            self.ax.plot(
                times[valid_f0],
                f0_freqs[valid_f0],
                color=color,
                linestyle="-",
                linewidth=max(1.0, linewidth * 0.65),
                alpha=min(0.85, alpha + 0.05),
                label=f"{label} F0",
            )

    def _plot_h1_and_f0_elelet(self, freq_to_channel_idx):
        """Plot each visible CSV H1 contour plus its derived F0 = H1 / 2."""
        for _, label, times, h1_freqs, color, linestyle, linewidth, alpha in self._contour_series():
            valid_h1 = h1_freqs > 0
            if not np.any(valid_h1):
                continue

            time_idx = times * self.sr / self.elelet_transform.stride
            h1_freq_idx = np.array([freq_to_channel_idx(f) for f in h1_freqs])
            self.ax.plot(
                time_idx[valid_h1],
                h1_freq_idx[valid_h1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=f"{label} H1",
            )

            f0_freqs = h1_freqs / 2.0
            valid_f0 = f0_freqs > 0
            f0_freq_idx = np.array([freq_to_channel_idx(f) for f in f0_freqs])
            self.ax.plot(
                time_idx[valid_f0],
                f0_freq_idx[valid_f0],
                color=color,
                linestyle="-",
                linewidth=max(1.5, linewidth * 0.65),
                alpha=min(0.85, alpha + 0.05),
                label=f"{label} F0",
            )

    def _plot_upper_harmonics_stft(self):
        """Plot harmonic guides for every enabled contour, omitting legend entries."""
        for _, _, times, h1_freqs, color, _, _, alpha in self._contour_series(require_master_toggle=False):
            fundamental_freqs = h1_freqs / 2.0
            for harmonic in range(2, self.max_harmonic + 1):
                freqs = fundamental_freqs * harmonic
                valid = (fundamental_freqs > 0) & (freqs <= self.fmax)
                if not np.any(valid):
                    continue
                self.ax.plot(
                    times[valid],
                    freqs[valid],
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=min(0.6, alpha * 0.7),
                    label="_nolegend_",
                )

    def _plot_upper_harmonics_elelet(self, freq_to_channel_idx):
        """Plot harmonic guides for every enabled contour, omitting legend entries."""
        for _, _, times, h1_freqs, color, _, _, alpha in self._contour_series(require_master_toggle=False):
            fundamental_freqs = h1_freqs / 2.0
            time_idx = times * self.sr / self.elelet_transform.stride
            for harmonic in range(2, self.max_harmonic + 1):
                freqs = fundamental_freqs * harmonic
                valid = (fundamental_freqs > 0) & (freqs <= self.fmax)
                if not np.any(valid):
                    continue
                freq_idx = np.array([freq_to_channel_idx(f) for f in freqs])
                self.ax.plot(
                    time_idx[valid],
                    freq_idx[valid],
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=min(0.6, alpha * 0.7),
                    label="_nolegend_",
                )

    def update_plot(self):
        """Update the plot with current data"""
        self.ax.clear()

        audio_path = self.audio_files[self.current_idx]

        # Plot spectrogram based on current mode (compute on-demand)
        if self.spec_mode == 'stft':
            # Compute STFT if needed
            self.compute_stft()
            # STFT spectrogram
            freq_mask = (self.spec_freqs >= self.fmin) & (self.spec_freqs <= self.fmax)
            self.ax.pcolormesh(self.spec_times, self.spec_freqs[freq_mask],
                              self.S_db[freq_mask, :],
                              shading='gouraud', cmap='Greys', vmin=-80, vmax=0)
        else:
            # Compute Elelet if needed
            self.compute_elelet()
            # Elelet spectrogram - using Elelet's plot logic
            c = np.abs(self.elelet_coeffs)
            c = np.log10(c + 1e-10)  # Apply log scale

            # Filter to fmax (same logic as transform.plot)
            fc = self.elelet_transform.fc[self.elelet_transform.fc >= self.elelet_transform.fmin]
            if self.fmax is not None:
                freq_idx_max = np.argmax(fc > self.fmax)
                if freq_idx_max == 0:  # All frequencies below fmax
                    freq_idx_max = len(fc)
                c = c[:freq_idx_max, :]
                fc_filtered = fc[:freq_idx_max]
            else:
                fc_filtered = fc

            # Optimize color intensity for 15-75Hz region
            freq_mask_opt = (fc_filtered >= 15) & (fc_filtered <= 75)
            if np.any(freq_mask_opt):
                # Get data from the optimized frequency range
                c_opt = c[freq_mask_opt, :]
                # Use percentiles for robust vmin/vmax
                vmin = np.percentile(c_opt, 5)
                vmax = np.percentile(c_opt, 95)
            else:
                vmin = np.percentile(c, 5)
                vmax = np.percentile(c, 95)

            # Plot using pcolor (like in transform.plot)
            self.ax.pcolor(c, cmap='Greys', vmin=vmin, vmax=vmax)

            # Set y-axis ticks to show actual frequencies (like in transform.plot)
            locs = np.linspace(self.elelet_transform.fmin, c.shape[0] - 1, min(len(fc), 10)).astype(int)
            self.ax.set_yticks(locs)
            self.ax.set_yticklabels([int(np.round(fc[i])) for i in locs])

            # Set x-axis to show time in seconds
            num_time_labels = 10
            xticks = np.linspace(0, c.shape[1] - 1, num_time_labels)
            self.ax.set_xticks(xticks)
            self.ax.set_xticklabels(
                [np.round(x, 1) for x in np.linspace(0, len(self.audio) / self.sr, num_time_labels)]
            )

        # Plot f0 overlays (convert coordinates for Elelet mode)
        if self.spec_mode == 'elelet':
            # Convert frequency from Hz to channel indices
            # Note: when fmin > 0, coefficients only contain frequencies >= fmin
            fc = self.elelet_transform.fc
            fc_filtered = fc[fc >= self.elelet_transform.fmin]
            def freq_to_channel_idx(freq_hz):
                """Convert frequency in Hz to channel index"""
                if freq_hz <= 0:
                    return 0
                # Find nearest channel in the filtered frequency array
                idx = np.argmin(np.abs(fc_filtered - freq_hz))
                return idx

            if self.show_upper_harmonics:
                self._plot_upper_harmonics_elelet(freq_to_channel_idx)

            self._plot_h1_and_f0_elelet(freq_to_channel_idx)

            # Plot correction points (coral markers)
            if self.correction_points:
                times, freqs = zip(*self.correction_points)
                times_idx = np.array(times) * self.sr / self.elelet_transform.stride
                freqs_idx = np.array([freq_to_channel_idx(f) for f in freqs])
                self.ax.plot(times_idx, freqs_idx, 'o', markersize=6, label='Correction points',
                       markeredgewidth=2, markerfacecolor='darkturquoise', markeredgecolor='darkturquoise')

            # Plot start line (green vertical line)
            if self.start_point:
                start_time_idx = self.start_point[0] * self.sr / self.elelet_transform.stride
                self.ax.axvline(start_time_idx, color='green', linestyle='-',
                               linewidth=2, label='Start', alpha=0.8)

            # Plot end line (blue vertical line)
            if self.end_point:
                end_time_idx = self.end_point[0] * self.sr / self.elelet_transform.stride
                self.ax.axvline(end_time_idx, color='blue', linestyle='-',
                               linewidth=2, label='End', alpha=0.8)
        else:
            # STFT mode - use real coordinates
            if self.show_upper_harmonics:
                self._plot_upper_harmonics_stft()

            self._plot_h1_and_f0_stft()

            # Plot correction points (coral markers)
            if self.correction_points:
                times, freqs = zip(*self.correction_points)
                self.ax.plot(times, freqs, 'o', markersize=6, label='Correction points',
                       markeredgewidth=2, markerfacecolor='darkturquoise', markeredgecolor='darkturquoise')

            # Plot start line (green vertical line)
            if self.start_point:
                self.ax.axvline(self.start_point[0], color='green', linestyle='-',
                               linewidth=2, label='Start', alpha=0.8)

            # Plot end line (blue vertical line)
            if self.end_point:
                self.ax.axvline(self.end_point[0], color='blue', linestyle='-',
                               linewidth=2, label='End', alpha=0.8)

        self.ax.set_xlabel('Time (s)', fontsize=12)
        self.ax.set_ylabel('Frequency (Hz)', fontsize=12)

        # Add spectrogram mode and display state to title
        spec_mode_label = "STFT" if self.spec_mode == 'stft' else "ELELET"
        contours_label = self._contour_title_summary()
        harmonics_label = "ON" if self.show_upper_harmonics else "OFF"
        self.ax.set_title(f'[{self.current_idx+1}/{len(self.audio_files)}] {audio_path.name} | Mode: {spec_mode_label} | Contours: {contours_label} | Harmonics: {harmonics_label}',
                    fontsize=10, fontweight='bold')
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(False)

        # Set y-axis limits based on mode
        if self.spec_mode == 'stft':
            self.ax.set_ylim([self.fmin, self.fmax])
        # For Elelet mode, the y-limits are automatically set by the pcolor plot

        self.fig.canvas.draw()

    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return

        time = event.xdata
        freq = event.ydata
        if time is None or freq is None:
            return

        # Convert from index coordinates to real coordinates if in Elelet mode
        if self.spec_mode == 'elelet':
            # Convert time from frame index to seconds
            time = time * self.elelet_transform.stride / self.sr
            # Convert frequency from channel index to Hz
            # Note: when fmin > 0, coefficients only contain frequencies >= fmin
            fc = self.elelet_transform.fc
            fc_filtered = fc[fc >= self.elelet_transform.fmin]
            freq_idx = int(np.clip(freq, 0, len(fc_filtered) - 1))
            freq = fc_filtered[freq_idx]

        if self.marking_mode == 'start':
            # Mark start time (only x-coordinate matters)
            self.start_point = (time, freq)
            print(f"Marked START time: t={time:.3f}s")
            self.marking_mode = None  # Reset mode
            if self.start_point and self.end_point:
                self.apply_corrections()
            self.update_plot()

        elif self.marking_mode == 'end':
            # Mark end time (only x-coordinate matters)
            self.end_point = (time, freq)
            print(f"Marked END time: t={time:.3f}s")
            self.marking_mode = None  # Reset mode
            if self.start_point and self.end_point:
                self.apply_corrections()
            self.update_plot()

        elif event.button == 1:  # Left click - add correction point
            self.correction_points.append((time, freq))
            self.correction_points.sort(key=lambda x: x[0])  # Sort by time
            self.apply_corrections()
            self.update_plot()
            print(f"Added correction point: t={time:.3f}s, f0={freq:.1f}Hz")

        elif event.button == 3:  # Right click - remove nearest point
            # Prioritize correction points over start/end lines
            # First check if there are correction points nearby
            if self.correction_points:
                # Find nearest correction point
                distances = [(t - time)**2 + (f - freq)**2
                           for t, f in self.correction_points]
                nearest_idx = np.argmin(distances)
                nearest_dist = distances[nearest_idx]

                # If nearest correction point is reasonably close, remove it
                # Use a threshold that works for both time and frequency dimensions
                if nearest_dist < 1.0:  # Adjust threshold as needed
                    removed = self.correction_points.pop(nearest_idx)
                    self.apply_corrections()
                    self.update_plot()
                    print(f"Removed correction point: t={removed[0]:.3f}s, f0={removed[1]:.1f}Hz")
                    return

            # If no nearby correction points, check start/end lines
            if self.start_point:
                # Only check x-distance (time) for vertical lines
                dist_start = abs(self.start_point[0] - time)
                if dist_start < 0.5:  # Within 0.5 seconds
                    self.start_point = None
                    print("Removed START line")
                    self.apply_corrections()
                    self.update_plot()
                    return

            if self.end_point:
                # Only check x-distance (time) for vertical lines
                dist_end = abs(self.end_point[0] - time)
                if dist_end < 0.5:  # Within 0.5 seconds
                    self.end_point = None
                    print("Removed END line")
                    self.apply_corrections()
                    self.update_plot()
                    return

    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'right' or event.key == 'n':
            self.load_file(self.current_idx + 1)
        elif event.key == 'left' or event.key == 'p':
            self.load_file(self.current_idx - 1)
        elif event.key == ' ':  # Spacebar
            self.play_audio()
        elif event.key == 'r':
            self.toggle_spectrogram()
        elif event.key == 'w':
            self.set_marking_mode('start')
        elif event.key == 'e':
            self.set_marking_mode('end')
        elif event.key == 'c':
            self.clear_corrections()
        elif event.key == 'f':
            self.toggle_f0_contours()
        elif event.key == 'h':
            self.toggle_upper_harmonics()
        elif event.key in {'1', '2', '3', '4', '5', '6', '7', '8', '9'}:
            self.toggle_single_contour(event.key)
        elif event.key == 'q':
            self.quit_tool()

    def _remove_duplicate_times(self, times, freqs):
        """Remove duplicate time values by averaging their frequencies"""
        times_arr = np.array(times)
        freqs_arr = np.array(freqs)

        # Find unique times
        unique_times = np.unique(times_arr)
        unique_freqs = []

        for t in unique_times:
            # Find all frequencies at this time
            mask = times_arr == t
            # Average them
            avg_freq = np.mean(freqs_arr[mask])
            unique_freqs.append(avg_freq)

        return unique_times.tolist(), unique_freqs

    def apply_corrections(self):
        """Interpolate f0 based on correction points and/or start/end points using cubic spline"""
        # Start with original
        self.corrected_f0 = self.original_f0.copy()

        # Apply start/end boundaries (only sets f0=0 outside, doesn't affect curve inside)
        if self.start_point and self.end_point:
            t_start, _ = self.start_point  # Only use time, not frequency
            t_end, _ = self.end_point

            # Ensure start comes before end
            if t_start > t_end:
                t_start, t_end = t_end, t_start

            # Set f0 to 0 before start and after end (this is ALL start/end do)
            self.corrected_f0[self.f0_time < t_start] = 0.0
            self.corrected_f0[self.f0_time > t_end] = 0.0

        # Apply correction points interpolation (independent of start/end)
        if len(self.correction_points) > 0:
            if len(self.correction_points) == 1:
                # Single point - set constant value in neighborhood
                t, f = self.correction_points[0]
                # Find nearest frames within 0.5s
                mask = np.abs(self.f0_time - t) < 0.5
                self.corrected_f0[mask] = f

            elif len(self.correction_points) == 2:
                # Two points - use linear interpolation
                times, freqs = zip(*self.correction_points)
                # Remove duplicates if any
                times, freqs = self._remove_duplicate_times(times, freqs)
                if len(times) >= 2:
                    interp = interp1d(times, freqs, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
                    t_min, t_max = min(times), max(times)
                    mask = (self.f0_time >= t_min) & (self.f0_time <= t_max)
                    self.corrected_f0[mask] = interp(self.f0_time[mask])

            else:
                # Multiple points (3+) - use cubic spline for smooth interpolation
                times, freqs = zip(*self.correction_points)
                # Remove duplicates if any
                times, freqs = self._remove_duplicate_times(times, freqs)
                if len(times) >= 2:
                    # Need at least 2 points for interpolation
                    kind = 'cubic' if len(times) >= 4 else 'linear'
                    interp = interp1d(times, freqs, kind=kind,
                                     bounds_error=False, fill_value='extrapolate')
                    t_min, t_max = min(times), max(times)
                    mask = (self.f0_time >= t_min) & (self.f0_time <= t_max)
                    self.corrected_f0[mask] = interp(self.f0_time[mask])

        # Clip to valid range
        self.corrected_f0 = np.clip(self.corrected_f0, 0, self.fmax)

    def clear_corrections(self):
        """Clear all correction points and start/end lines"""
        self.correction_points = []
        self.start_point = None
        self.end_point = None
        self.corrected_f0 = self.original_f0.copy()
        self.update_plot()
        print("Cleared all correction points and start/end lines")

    def save_corrections(self, quiet=False):
        """Save corrected f0 to file"""
        audio_path = self.audio_files[self.current_idx]
        output_path = self.corrected_dir / f"{audio_path.stem}.f0.csv"

        # Mark start and end points
        start_mask = np.zeros(len(self.f0_time), dtype=int)
        end_mask = np.zeros(len(self.f0_time), dtype=int)

        # If start/end points are manually set, use them
        if self.start_point:
            idx = np.argmin(np.abs(self.f0_time - self.start_point[0]))
            start_mask[idx] = 1
        else:
            # Otherwise, use first non-zero f0 value
            nonzero_indices = np.where(self.corrected_f0 > 0)[0]
            if len(nonzero_indices) > 0:
                start_mask[nonzero_indices[0]] = 1

        if self.end_point:
            idx = np.argmin(np.abs(self.f0_time - self.end_point[0]))
            end_mask[idx] = 1
        else:
            # Otherwise, use last non-zero f0 value
            nonzero_indices = np.where(self.corrected_f0 > 0)[0]
            if len(nonzero_indices) > 0:
                end_mask[nonzero_indices[-1]] = 1

        # Save only the corrected contour and boundary markers.
        df = pd.DataFrame({
            'time': self.f0_time,
            'frequency': self.corrected_f0,
            'start_point': start_mask,
            'end_point': end_mask,
        })
        df.to_csv(output_path, index=False)

        if not quiet:
            print(f"✓ Saved corrections to: {output_path}")
            print(f"  Correction points: {len(self.correction_points)}")
            if self.start_point:
                print(f"  Start time: t={self.start_point[0]:.3f}s")
            elif len(np.where(self.corrected_f0 > 0)[0]) > 0:
                print(f"  Start time (auto): t={self.f0_time[np.where(self.corrected_f0 > 0)[0][0]]:.3f}s")
            if self.end_point:
                print(f"  End time: t={self.end_point[0]:.3f}s")
            elif len(np.where(self.corrected_f0 > 0)[0]) > 0:
                print(f"  End time (auto): t={self.f0_time[np.where(self.corrected_f0 > 0)[0][-1]]:.3f}s")

            # Update title to show saved status
            title = self.ax.get_title()
            if '✓ SAVED' not in title:
                self.ax.set_title(title.split('\n')[0] + ' ✓ SAVED\n' + '\n'.join(title.split('\n')[1:]),
                            fontsize=11, fontweight='bold', color='green')
                self.fig.canvas.draw()

    def quit_tool(self):
        """Save current file and quit"""
        print("\nSaving current file before quitting...")
        self.save_corrections(quiet=False)
        print("Goodbye!")
        plt.close('all')

    def run(self):
        """Start the interactive tool"""
        print("="*60)
        print("INTERACTIVE F0 CORRECTION TOOL (with ELELET)")
        print("="*60)
        print(f"Files: {len(self.audio_files)}")
        print(f"F0 directory: {self.f0_dir}")
        print(f"Output directory: {self.corrected_dir}")
        if self.use_precomputed:
            print(f"Using pre-computed representations (FAST)")
            print(f"  STFT: {self.stft_dir}")
            print(f"  Elelet: {self.elelet_dir}")
        else:
            print(f"Computing representations on-the-fly (SLOW)")
        print("\nControls:")
        print("  LEFT CLICK: Add correction point")
        print("  RIGHT CLICK: Remove nearest point")
        print("  SPACEBAR: Play audio sample")
        print("  R: Switch spectrogram representation (STFT ↔ ELELET)")
        print("  W + CLICK: Mark start time (green line)")
        print("  E + CLICK: Mark end time (blue line)")
        print("    → Start/End define time boundaries only (f0=0 outside)")
        print("    → Add correction points within region for interpolation")
        print("  F: Toggle all F0 contour overlays")
        print("  1/2/3...: Toggle individual contours shown in the right panel")
        print("  H: Toggle harmonic overlays for enabled contours (base F0 = H1 / 2)")
        print("  Arrow keys / N/P: Next/Previous file (auto-saves)")
        print("  C: Clear all points")
        print("  Q: Quit (saves and exits)")
        print("\nSpectrograms:")
        print("  - STFT: Standard Short-Time Fourier Transform")
        print("  - ELELET: Elephant Wavelet Transform (custom filterbank)")
        print("\nAuto-save:")
        print("  - Automatically saves when navigating to next/previous file")
        print("  - Automatically saves when quitting")
        print("\nInterpolation:")
        print("  - Start/End lines only mark time boundaries (not f0 values)")
        print("  - f0 = 0 before start and after end")
        print("  - Cubic spline interpolates between correction points")
        print("="*60 + "\n")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive tool for manually correcting f0 by clicking on spectrograms (STFT + Elelet)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Correct contours from one extracted algorithm directory
  python f0_extraction/annotate_f0.py --input data/rumbles --f0_dir f0_elelet

  # With custom frequency range
  python f0_extraction/annotate_f0.py --input data/rumbles --f0_dir f0_stft --fmin 0 --fmax 80

  # Start at sample number 5 (6th file, 0-indexed)
  python f0_extraction/annotate_f0.py --input data/rumbles --f0_dir f0_elelet --start 5

Usage:
  1. Press 'R' to toggle between STFT and Elelet spectrograms
  2. Press 'W', click on spectrogram to mark START time (green line)
  3. Press 'E', click on spectrogram to mark END time (blue line)
     → f0 = 0 before start and after end
  4. Add correction points (left click) within the region for f0 values
  5. Tool uses cubic spline interpolation between correction points
  6. Right-click to remove nearby points
  7. Press 'F' to hide/show all F0 contour overlays
  8. Press '1', '2', '3'... to toggle individual contours in the right panel
  9. Press 'H' to hide/show harmonic overlays for enabled contours (base F0 = H1 / 2)
  10. Use arrow keys to navigate between files (auto-saves)
  11. Press 'Q' to quit (auto-saves and exits)
  12. Corrected f0 saved to data/rumbles/f0_corrected/

Spectrograms:
  - STFT: Standard Short-Time Fourier Transform
  - Elelet: Elephant Wavelet Transform with custom filterbank

Note: Files are automatically saved when you navigate to the next file or quit.
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with audio files and an extracted f0_* subdirectory')
    parser.add_argument('--algorithm_name', type=str, default=None,
                        help='Algorithm name used to infer f0_<algorithm> when --f0_dir is omitted')
    parser.add_argument('--f0_dir', type=str, default=None,
                        help='Explicit input contour directory under --input')
    parser.add_argument('--corrected_dir', type=str, default=None,
                        help='Explicit corrected contour output directory under --input')
    parser.add_argument('--stft_dir', type=str, default=None,
                        help='Explicit precomputed STFT directory under --input')
    parser.add_argument('--elelet_dir', type=str, default=None,
                        help='Explicit precomputed Elelet directory under --input')
    parser.add_argument('--initial_spec_mode', choices=('elelet', 'stft'), default='stft',
                        help='Initial spectrogram view')
    parser.add_argument('--representation_fmax', type=float, default=750,
                        help='Frequency label used for default precomputed representation directories')
    parser.add_argument('--enable_elelet_view', action=argparse.BooleanOptionalAction, default=True,
                        help='Allow switching to the Elelet spectrogram view')
    parser.add_argument('--max_harmonic', type=int, default=10,
                        help='Highest harmonic to draw when harmonic overlays are enabled')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate (default: 16000)')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution (default: 0.016)')
    parser.add_argument('--n_fft', type=int, default=8192,
                        help='FFT size for spectrogram (default: 8192)')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop length for spectrogram (default: 256)')
    parser.add_argument('--fmin', type=float, default=10,
                        help='Minimum frequency to display (Hz, default: 0)')
    parser.add_argument('--fmax', type=float, default=200,
                        help='Maximum frequency to display (Hz, default: 100)')
    parser.add_argument('--start', type=int, default=0,
                        help='Sample number to start at (0-indexed, default: 0)')
    parser.add_argument('--precomputed', action='store_true', default=True,
                        help='Use pre-computed representations (default: True)')
    parser.add_argument('--no_precomputed', dest='precomputed', action='store_false',
                        help='Compute representations on-the-fly (slower)')

    args = parser.parse_args()

    corrector = F0Corrector(
        audio_dir=args.input,
        sr=args.sr,
        hop_length=args.hop_length,
        frame_resolution=args.frame_resolution,
        n_fft=args.n_fft,
        fmin=args.fmin,
        fmax=args.fmax,
        start_idx=args.start,
        use_precomputed=args.precomputed,
        algorithm_name=args.algorithm_name,
        f0_dir=args.f0_dir,
        corrected_dir=args.corrected_dir,
        stft_dir=args.stft_dir,
        elelet_dir=args.elelet_dir,
        initial_spec_mode=args.initial_spec_mode,
        representation_fmax=args.representation_fmax,
        enable_elelet_view=args.enable_elelet_view,
        max_harmonic=args.max_harmonic,
    )

    corrector.run()


if __name__ == "__main__":
    main()
