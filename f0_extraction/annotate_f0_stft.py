"""
Interactive f0 correction tool
Click on spectrogram to add correction points, interpolates between clicks
"""

import argparse
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import soundfile as sf


class F0Corrector:
    def __init__(self, audio_dir, sr=16000, frame_resolution=0.016,
                 n_fft=8192, fmin=0, fmax=100, start_idx=0):
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.frame_resolution = frame_resolution
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = 256

        # Find f0 directory
        self.f0_dir = self.audio_dir / f"f0_{frame_resolution:.3f}"
        if not self.f0_dir.exists():
            raise ValueError(f"F0 directory not found: {self.f0_dir}")

        # Find all audio files
        self.audio_files = sorted(list(self.audio_dir.glob("*.wav")))
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {self.audio_dir}")

        # Validate start index
        if start_idx < 0 or start_idx >= len(self.audio_files):
            print(f"Warning: start index {start_idx} out of range [0, {len(self.audio_files)-1}], using 0")
            start_idx = 0

        # Track corrected files
        self.corrected_dir = self.audio_dir / "f0_corrected"
        self.corrected_dir.mkdir(exist_ok=True)

        # Current state
        self.current_idx = 0
        self.correction_points = []  # List of (time, freq) tuples
        self.start_point = None  # (time, freq) for start of region
        self.end_point = None  # (time, freq) for end of region
        self.original_f0 = None
        self.corrected_f0 = None

        # Setup figure (single plot for spectrogram only)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Add buttons
        self.setup_buttons()

        # Load file at start index
        self.load_file(start_idx)

    def setup_buttons(self):
        """Setup navigation and action buttons"""
        # Previous button
        ax_prev = plt.axes([0.05, 0.02, 0.08, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous (←)')
        self.btn_prev.on_clicked(lambda e: self.load_file(self.current_idx - 1))

        # Next button
        ax_next = plt.axes([0.14, 0.02, 0.08, 0.04])
        self.btn_next = Button(ax_next, 'Next (→)')
        self.btn_next.on_clicked(lambda e: self.load_file(self.current_idx + 1))

        # Mark Start button
        ax_start = plt.axes([0.23, 0.02, 0.10, 0.04])
        self.btn_start = Button(ax_start, 'Start Line (W)')
        self.btn_start.on_clicked(lambda e: self.set_marking_mode('start'))

        # Mark End button
        ax_end = plt.axes([0.34, 0.02, 0.10, 0.04])
        self.btn_end = Button(ax_end, 'End Line (E)')
        self.btn_end.on_clicked(lambda e: self.set_marking_mode('end'))

        # Clear corrections button
        ax_clear = plt.axes([0.45, 0.02, 0.12, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear Points (C)')
        self.btn_clear.on_clicked(lambda e: self.clear_corrections())

        # Reset to original button
        ax_reset = plt.axes([0.58, 0.02, 0.12, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset Original (R)')
        self.btn_reset.on_clicked(lambda e: self.reset_to_original())

        # Quit button
        ax_quit = plt.axes([0.71, 0.02, 0.08, 0.04])
        self.btn_quit = Button(ax_quit, 'Quit (Q)')
        self.btn_quit.on_clicked(lambda e: self.quit_tool())

        # Marking mode
        self.marking_mode = None  # None, 'start', or 'end'

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
        self.f0_conf = df['confidence'].values

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
                    idx = np.where(start_mask)[0][0]
                    self.start_point = (df_corrected.loc[idx, 'time'],
                                       df_corrected.loc[idx, 'frequency'])
                else:
                    self.start_point = None
            else:
                self.start_point = None
            if 'end_point' in df_corrected.columns:
                end_mask = df_corrected['end_point'] == 1
                if end_mask.any():
                    idx = np.where(end_mask)[0][0]
                    self.end_point = (df_corrected.loc[idx, 'time'],
                                     df_corrected.loc[idx, 'frequency'])
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

        # Compute spectrogram
        D = librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
        self.S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        self.spec_times = librosa.times_like(self.S_db, sr=self.sr, hop_length=self.hop_length)
        self.spec_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # Update display
        self.update_plot()

    def update_plot(self):
        """Update the plot with current data"""
        self.ax.clear()

        audio_path = self.audio_files[self.current_idx]

        # Spectrogram with f0 overlay
        freq_mask = (self.spec_freqs >= self.fmin) & (self.spec_freqs <= self.fmax)
        self.ax.pcolormesh(self.spec_times, self.spec_freqs[freq_mask],
                          self.S_db[freq_mask, :],
                          shading='gouraud', cmap='magma', vmin=-80, vmax=0)

        # Plot original f0 (gray dashed)
        valid_orig = self.original_f0 > 0
        self.ax.plot(self.f0_time[valid_orig], self.original_f0[valid_orig],
                'gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Original f0')

        # Plot corrected f0 (cyan solid)
        valid_corr = self.corrected_f0 > 0
        self.ax.plot(self.f0_time[valid_corr], self.corrected_f0[valid_corr],
                'c-', linewidth=2, label='Corrected f0', alpha=0.8)

        # Plot correction points (red markers)
        if self.correction_points:
            times, freqs = zip(*self.correction_points)
            self.ax.plot(times, freqs, 'ro', markersize=6, label='Correction points',
                   markeredgewidth=2, markerfacecolor='red')

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
        self.ax.set_title(f'[{self.current_idx+1}/{len(self.audio_files)}] {audio_path.name}\n'
                    f'CLICK: add correction point | RIGHT CLICK: remove | '
                    f'S+CLICK: start line (green) | E+CLICK: end line (blue) | ←→: navigate (auto-saves) | C: clear | R: reset | Q: quit',
                    fontsize=10, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim([self.fmin, self.fmax])

        plt.tight_layout()
        self.fig.canvas.draw()

    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return

        time = event.xdata
        freq = event.ydata
        if time is None or freq is None:
            return

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
            # Check if clicking near start/end lines first
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

            # Remove nearest correction point
            if not self.correction_points:
                return

            # Find nearest correction point
            distances = [(t - time)**2 + (f - freq)**2
                       for t, f in self.correction_points]
            nearest_idx = np.argmin(distances)
            removed = self.correction_points.pop(nearest_idx)
            self.apply_corrections()
            self.update_plot()
            print(f"Removed correction point: t={removed[0]:.3f}s, f0={removed[1]:.1f}Hz")

    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'right' or event.key == 'n':
            self.load_file(self.current_idx + 1)
        elif event.key == 'left' or event.key == 'p':
            self.load_file(self.current_idx - 1)
        elif event.key == 'w':
            self.set_marking_mode('start')
        elif event.key == 'e':
            self.set_marking_mode('end')
        elif event.key == 'c':
            self.clear_corrections()
        elif event.key == 'r':
            self.reset_to_original()
        elif event.key == 'q':
            self.quit_tool()

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
                interp = interp1d(times, freqs, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                t_min, t_max = min(times), max(times)
                mask = (self.f0_time >= t_min) & (self.f0_time <= t_max)
                self.corrected_f0[mask] = interp(self.f0_time[mask])

            else:
                # Multiple points (3+) - use cubic spline for smooth interpolation
                times, freqs = zip(*self.correction_points)
                interp = interp1d(times, freqs, kind='cubic',
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

    def reset_to_original(self):
        """Reset to original f0"""
        self.correction_points = []
        self.start_point = None
        self.end_point = None
        self.corrected_f0 = self.original_f0.copy()
        self.update_plot()
        print("Reset to original f0")

    def save_corrections(self, quiet=False):
        """Save corrected f0 to file"""
        audio_path = self.audio_files[self.current_idx]
        output_path = self.corrected_dir / f"{audio_path.stem}.f0.csv"

        # Mark start and end points
        start_mask = np.zeros(len(self.f0_time), dtype=int)
        end_mask = np.zeros(len(self.f0_time), dtype=int)

        if self.start_point:
            idx = np.argmin(np.abs(self.f0_time - self.start_point[0]))
            start_mask[idx] = 1

        if self.end_point:
            idx = np.argmin(np.abs(self.f0_time - self.end_point[0]))
            end_mask[idx] = 1

        # Save to CSV (only time, frequency, start_point, end_point)
        df = pd.DataFrame({
            'time': self.f0_time,
            'frequency': self.corrected_f0,
            'start_point': start_mask,
            'end_point': end_mask
        })
        df.to_csv(output_path, index=False)

        if not quiet:
            print(f"✓ Saved corrections to: {output_path}")
            print(f"  Correction points: {len(self.correction_points)}")
            if self.start_point:
                print(f"  Start time: t={self.start_point[0]:.3f}s")
            if self.end_point:
                print(f"  End time: t={self.end_point[0]:.3f}s")

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
        print("INTERACTIVE F0 CORRECTION TOOL")
        print("="*60)
        print(f"Files: {len(self.audio_files)}")
        print(f"F0 directory: {self.f0_dir}")
        print(f"Output directory: {self.corrected_dir}")
        print("\nControls:")
        print("  LEFT CLICK: Add correction point")
        print("  RIGHT CLICK: Remove nearest point")
        print("  S + CLICK: Mark start time (green line)")
        print("  E + CLICK: Mark end time (blue line)")
        print("    → Start/End define time boundaries only (f0=0 outside)")
        print("    → Add correction points within region for interpolation")
        print("  Arrow keys / N/P: Next/Previous file (auto-saves)")
        print("  C: Clear all points")
        print("  R: Reset to original f0")
        print("  Q: Quit (saves and exits)")
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
        description='Interactive tool for manually correcting f0 by clicking on spectrograms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Correct f0 for all files in directory
  python correct_f0_interactive.py --input data/rumbles

  # With custom frequency range
  python correct_f0_interactive.py --input data/rumbles --fmin 0 --fmax 80

  # Start at sample number 5 (6th file, 0-indexed)
  python correct_f0_interactive.py --input data/rumbles --start 5

Usage:
  1. Press 'W', click on spectrogram to mark START time (green line)
  2. Press 'E', click on spectrogram to mark END time (blue line)
     → f0 = 0 before start and after end
  3. Add correction points (left click) within the region for f0 values
  4. Tool uses cubic spline interpolation between correction points
  5. Right-click to remove nearby points
  6. Use arrow keys to navigate between files (auto-saves)
  7. Press 'Q' to quit (auto-saves and exits)
  8. Corrected f0 saved to data/rumbles/f0_corrected/

Note: Files are automatically saved when you navigate to the next file or quit.
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with audio files and f0_X.XXX/ subdirectory')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate (default: 16000)')
    parser.add_argument('--frame_resolution', type=float, default=0.016,
                        help='Frame resolution (default: 0.016)')
    parser.add_argument('--n_fft', type=int, default=8192,
                        help='FFT size for spectrogram (default: 8192)')
    parser.add_argument('--fmin', type=float, default=0,
                        help='Minimum frequency to display (Hz, default: 0)')
    parser.add_argument('--fmax', type=float, default=100,
                        help='Maximum frequency to display (Hz, default: 100)')
    parser.add_argument('--start', type=int, default=0,
                        help='Sample number to start at (0-indexed, default: 0)')

    args = parser.parse_args()

    corrector = F0Corrector(
        audio_dir=args.input,
        sr=args.sr,
        frame_resolution=args.frame_resolution,
        n_fft=args.n_fft,
        fmin=args.fmin,
        fmax=args.fmax,
        start_idx=args.start
    )

    corrector.run()


if __name__ == "__main__":
    main()
