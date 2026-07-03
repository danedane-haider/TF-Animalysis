import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path

from .utils_auditory_scales import audfilters
from .utils_elelet import (
    _numpy_kernel_fft,
    _torch_kernel_fft,
    _validate_fft_backend,
    circ_conv,
    circ_conv_numpy,
)
from .utils_elespectrogram import melscale_fbanks
from .utils_harmonic_mask import load_f0_csv, masked_spectrogram


#################################    ELELET TRANSFORM    #################################

class Elelet:
    """
    Compute an elephant filterbank transform of an audio signal.

    Args:
        kernel_size: Size of the convolution kernel (default: 8192)
        num_channels: Number of filter channels (default: 1024)
        stride: Downsampling factor for convolution (default: 320)
        f_min: Minimum frequency in Hz (default: 5)
        f_max: Maximum frequency in Hz (default: 1000)
        fs: Sample rate in Hz (default: 16000)
        supp_mult: Support multiplier (default: 0.2)
        scale: Frequency scale ('elelog' or other) (default: 'elelog')
        pad_mode: Boundary mode for convolution. Use 'circular' for wraparound
            or 'reflect' to mirror-pad like torch.stft(center=True).
        backend: FFT implementation. ``'fft'`` computes the full convolution;
            ``'fft_decimated'`` evaluates only samples selected by ``stride``.
        channel_batch_size: Optional number of filter channels to process at a
            time. This bounds intermediate memory for batched ML inputs.
        cache_kernel_fft: Cache the most recently used kernel spectra. This is
            useful for repeated, fixed-size ML batches.
    """

    def __init__(
        self,
        kernel_size=8192,
        num_channels=1024,
        stride=320,
        f_min=5,
        f_max=1000,
        fs=16000,
        supp_mult=0.2,
        scale='elelog',
        use_torch=False,
        pad_mode="circular",
        backend="fft_decimated",
        channel_batch_size=None,
        cache_kernel_fft=True,
    ):

        _validate_fft_backend(backend)
        if channel_batch_size is not None and channel_batch_size <= 0:
            raise ValueError("channel_batch_size must be positive or None.")

        [kernels, _, fc, _, _] = audfilters(
            kernel_size=kernel_size,
            num_channels=num_channels,
            fmin=f_min,
            fmax=f_max,
            fs=fs,
            supp_mult=supp_mult,
            scale=scale,
        )

        if use_torch:
            self.kernels = kernels
            self.fc = fc
        else:
            self.kernels = kernels.numpy()
            self.fc = fc.numpy()
        self.stride = stride
        self.fs = fs
        self.num_channels = num_channels
        self.f_max = f_max
        self.f_min = f_min
        self.use_torch = use_torch
        self.pad_mode = pad_mode
        self.backend = backend
        self.channel_batch_size = channel_batch_size
        self.cache_kernel_fft = bool(cache_kernel_fft)
        self._kernel_fft_cache_key = None
        self._kernel_fft_cache = None

    def clear_fft_cache(self):
        """Discard cached kernel spectra, for example after changing kernels."""
        self._kernel_fft_cache_key = None
        self._kernel_fft_cache = None

    def _convolution_length(self, audio, kernel_size):
        length = audio.shape[-1]
        if self.pad_mode != "circular":
            length += 2 * (kernel_size // 2)
        return max(length, kernel_size)

    def _get_kernel_fft(self, audio, kernels):
        if not self.cache_kernel_fft:
            return None

        length = self._convolution_length(audio, kernels.shape[-1])
        if self.use_torch:
            key = (
                "torch",
                length,
                tuple(kernels.shape),
                kernels.dtype,
                audio.device,
            )
            if key != self._kernel_fft_cache_key:
                self._kernel_fft_cache = _torch_kernel_fft(
                    kernels,
                    length,
                    audio.device,
                )
                self._kernel_fft_cache_key = key
        else:
            key = (
                "numpy",
                length,
                tuple(kernels.shape),
                kernels.dtype.str,
            )
            if key != self._kernel_fft_cache_key:
                self._kernel_fft_cache = _numpy_kernel_fft(kernels, length)
                self._kernel_fft_cache_key = key
        return self._kernel_fft_cache

    def forward(self, audio):
        if self.f_min > 0:
            # only compute channels above f_min
            if self.use_torch:
                idx_start = torch.argmax((self.fc >= self.f_min).to(torch.int)).item()
            else:
                idx_start = np.argmax(self.fc >= self.f_min)
            kernels_to_use = self.kernels[idx_start:, :]
        else:
            kernels_to_use = self.kernels
        kernel_fft = self._get_kernel_fft(audio, kernels_to_use)
        if self.use_torch:
            coeffs = circ_conv(
                audio,
                kernels_to_use,
                d=self.stride,
                pad_mode=self.pad_mode,
                backend=self.backend,
                kernel_fft=kernel_fft,
                channel_batch_size=self.channel_batch_size,
            )
        else:
            coeffs = circ_conv_numpy(
                audio,
                kernels_to_use,
                d=self.stride,
                pad_mode=self.pad_mode,
                backend=self.backend,
                kernel_fft=kernel_fft,
                channel_batch_size=self.channel_batch_size,
            )
        return coeffs

    def __call__(self, audio):
        return self.forward(audio)

    def plot(
        self,
        c,
        L,
        f_max=None,
        log_scale=False,
        vmin=None,
        cmap="magma",
        figsize=(10, 6),
    ):
        """Plot filterbank representation.

        Creates a spectrogram-like visualization with frequency on y-axis and time on x-axis.
        Supports logarithmic scaling and frequency range limitation for better visualization.
        This version returns the figure object instead of displaying it directly.

        Args:
            c: Filterbank coefficients of shape (num_channels, num_frames)
            L (int, optional): Original signal length for time axis scaling. Default: None
            f_max (float, optional): Maximum frequency to display in Hz. Default: None
            log_scale (bool): Whether to apply log10 scaling to coefficients. Default: False
            vmin (float, optional): Minimum value for dynamic range clipping. Default: None
            cmap (str): Matplotlib colormap name. Default: 'inferno'
            figsize (tuple): Figure size as (width, height). Default: (10, 6)
            title (str, optional): Title for the plot. Default: None

        Returns:
            plt.Figure: Matplotlib figure object containing the plot

        Example:
            >>> transform = Elelet(num_channels=1024, f_max=100, fs=16000)
            >>> coeffs = transform(audio)
            >>> fig = transform.plot_ISACgram(coeffs, L=len(audio), log_scale=True)
            >>> plt.show()
        """
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        c = c.cpu().detach().squeeze().numpy() if isinstance(c, torch.Tensor) else c
        fc = self.fc.numpy() if isinstance(self.fc, torch.Tensor) else self.fc

        c = np.abs(c)

        if log_scale:
            c = np.log10(c + 1e-10)

        if f_max is None:
            c = c[: np.argmax(fc > self.f_max), :]
        else:
            c = c[: np.argmax(fc > f_max), :]

        if vmin is not None:
            ax.pcolor(c, cmap=cmap, vmin=np.min(c) * vmin)
        else:
            ax.pcolor(c, cmap=cmap)

        fc_fmin = self.fc[fc > self.f_min]
        locs = np.linspace(self.f_min, c.shape[0] - 1, min(len(fc_fmin), 10)).astype(int)
        ax.set_yticks(locs)
        ax.set_yticklabels([int(np.round(fc_fmin[i])) for i in locs])

        if L is not None:
            num_time_labels = 10
            xticks = np.linspace(0, c.shape[1] - 1, num_time_labels)
            ax.set_xticks(xticks)
            ax.set_xticklabels(
                [np.round(x, 1) for x in np.linspace(0, L / self.fs, num_time_labels)]
            )
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xlabel("Time samples")

        ax.set_ylabel("Frequency [Hz]")

        plt.tight_layout()
        return fig
    

##################################    ELE SPECTROGRAM & ELECC    #################################

class EleSpectrogram(torch.nn.Module):
    """Compute an EleSpectrogram.

    Analog to torchaudio.transforms.MelSpectrogram but with a custom
    auditory scale (like 'elelog' for elephant hearing) instead of just mel scale.

    Args:
        fs (int): Sample rate of audio signal. Default: 16000
        n_fft (int): Size of FFT. Default: 400
        win_length (int, optional): Window size. Default: n_fft
        hop_length (int, optional): Length of hop between STFT windows. Default: win_length // 2
        f_min (float): Minimum frequency. Default: 0.0
        f_max (float, optional): Maximum frequency. Default: fs / 2.0
        num_channels (int): Number of mel filterbanks. Default: 128
        window_fn (callable, optional): Window function. Default: torch.hann_window
        power (float): Exponent for magnitude spectrogram. Default: 2.0
        normalized (bool): Whether to normalize by window power. Default: False
        center (bool): Whether to pad waveform on both sides. Default: True
        pad_mode (str): Padding mode. Default: "reflect"
        onesided (bool): Whether to return onesided FFT. Default: True
        norm (str, optional): Mel filterbank normalization. Default: None
        scale (str): Auditory scale to use. Default: 'elelog'

    Example:
        >>> mel_spec = EleSpectrogram(fs=16000, n_fft=512, num_channels=128, scale='elelog')
        >>> waveform = torch.randn(1, 16000)
        >>> mel = mel_spec(waveform)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 8192,
        win_length: int = 8192,
        hop_length: int = 320,
        f_min: float = 5.0,
        f_max: float = 1000,
        n_mels: int = 1024,
        window_fn = None,
        power: float = 1.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        norm: str = None,
        scale: str = 'elelog',
        drop_empty_filters: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_mels = n_mels
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.norm = norm
        self.scale = scale
        self.drop_empty_filters = bool(drop_empty_filters)

        if window_fn is None:
            self.window = torch.hann_window(self.win_length)
        else:
            self.window = window_fn(self.win_length)

        n_freqs = n_fft // 2 + 1 if onesided else n_fft
        self.mel_filterbank = melscale_fbanks(
            n_freqs=n_freqs,
            fmin=self.f_min,
            fmax=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm=self.norm,
            scale=self.scale,
            drop_empty_filters=self.drop_empty_filters,
        )

        # Update num_channels to reflect actual number of filters (may be less due to zero removal)
        self.n_mels = self.mel_filterbank.shape[1]

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )

    def _power_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        spec = torch.abs(spec)
        if self.power != 1.0:
            spec = spec ** self.power
        return spec

    def _apply_filterbank(self, spec: torch.Tensor) -> torch.Tensor:
        mel_filterbank = self.mel_filterbank.to(device=spec.device, dtype=spec.dtype)
        return torch.matmul(spec.transpose(-2, -1), mel_filterbank).transpose(-2, -1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._apply_filterbank(self._power_spectrogram(self._stft(waveform)))
    

class EleCC(torch.nn.Module):
    """Computes EleCC.

    Analog to torchaudio.transforms.MFCC but with a custom
    auditory scales (like 'elelog' for elephant hearing).

    Args:
        fs (int): Sample rate of audio signal. Default: 16000
        n_mfcc (int): Number of EleCC coefficients. Default: 40
        dct_type (int): Type of DCT to use (2, 3, or 4). Default: 2
        norm (str): Norm to use for DCT. Default: "ortho"
        log_mels (bool): Whether to use log-mel. Default: False
        melkwargs (dict, optional): Keyword args for EleSpectrogram. Default: None

    Example:
        >>> mfcc_transform = EleCC(fs=16000, n_mfcc=40, melkwargs={"scale": "elelog", "num_channels": 128})
        >>> waveform = torch.randn(1, 16000)
        >>> mfcc = mfcc_transform(waveform)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        melkwargs: dict = None,
    ):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_mels = log_mels

        if melkwargs is None:
            melkwargs = {}
        self.mel_spectrogram = EleSpectrogram(sample_rate=sample_rate, **melkwargs)

        n_mels = self.mel_spectrogram.n_mels
        self.dct_mat = self._create_dct_matrix(n_mfcc, n_mels, dct_type, norm)

    def _create_dct_matrix(self, n_mfcc: int, n_mels: int, dct_type: int, norm: str) -> torch.Tensor:
        """Create DCT transformation matrix."""
        n = torch.arange(n_mels, dtype=torch.float32)
        k = torch.arange(n_mfcc, dtype=torch.float32).unsqueeze(1)

        if dct_type == 2:
            dct = torch.cos(torch.pi / n_mels * (n + 0.5) * k)
            if norm == "ortho":
                dct[0] *= 1.0 / torch.sqrt(torch.tensor(2.0))
                dct *= torch.sqrt(torch.tensor(2.0 / n_mels))
        else:
            raise ValueError(f"DCT type {dct_type} not supported. Only type 2 is currently implemented.")

        return dct

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): Input waveform of shape (..., time)

        Returns:
            torch.Tensor: EleCC features of shape (..., n_mfcc, time)
        """
        mel_spec = self.mel_spectrogram(waveform)

        if self.log_mels:
            mel_spec = torch.log(mel_spec + 1e-10)

        dct_mat = self.dct_mat.to(waveform.device)
        mfcc = torch.matmul(dct_mat, mel_spec)

        return mfcc


class MaskedEleSpectrogram(EleSpectrogram):
    """
    EleSpectrogram variant that applies the mel filterbank to a harmonic-masked
    spectrogram (instead of the raw spectrogram).
    """

    def forward(
        self,
        waveform: torch.Tensor,
        f0_hz: torch.Tensor | np.ndarray | str | Path,
        width_bins: int = 1,
        n_harmonics: int = 32,
        kernel_floor: float = 1e-3,
        return_mask: bool = False,
    ):
        if isinstance(f0_hz, (str, Path)):
            f0_hz = load_f0_csv(f0_hz)

        masked_spec, mask = masked_spectrogram(
            waveform=waveform,
            f0_hz=f0_hz,
            width_bins=width_bins,
            n_harmonics=n_harmonics,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
            normalized=self.normalized,
            center=self.center,
            pad_mode=self.pad_mode,
            onesided=self.onesided,
            kernel_floor=kernel_floor,
            return_mask=True,
        )
        mel_spec = self._apply_filterbank(masked_spec)

        if return_mask:
            return mel_spec, mask
        return mel_spec


class MaskedEleCC(MaskedEleSpectrogram, EleCC):
    def __init__(
        self,
        sample_rate=16000,
        n_mfcc=40,
        dct_type=2,
        norm="ortho",
        log_mels=True,
        melkwargs=None,
    ):
        if melkwargs is None:
            melkwargs = {}

        MaskedEleSpectrogram.__init__(
            self,
            sample_rate=sample_rate,
            **melkwargs,
        )

        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_mels = log_mels
        self.dct_mat = EleCC._create_dct_matrix(
            self,
            n_mfcc=n_mfcc,
            n_mels=self.n_mels,
            dct_type=dct_type,
            norm=norm,
        )

    def forward(self, waveform, f0_hz, **mask_kwargs):
        mel_spec = MaskedEleSpectrogram.forward(
            self,
            waveform,
            f0_hz,
            **mask_kwargs,
        )

        if self.log_mels:
            mel_spec = torch.log(mel_spec + 1e-10)

        dct_mat = self.dct_mat.to(device=mel_spec.device, dtype=mel_spec.dtype)
        return torch.matmul(dct_mat, mel_spec)
