import csv
from pathlib import Path

import torch
import torch.nn.functional as F

_KERNEL_CACHE: dict[tuple[int, int, int, float], torch.Tensor] = {}


def load_f0_csv(path: str | Path, column: str | None = None) -> torch.Tensor:
    """Load a one-column f0 contour from a CSV."""
    path = Path(path).expanduser()
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")

        if column is None:
            normalized = {name.strip().lower(): name for name in reader.fieldnames}
            for candidate in ("f0_hz", "f0", "frequency"):
                if candidate in normalized:
                    column = normalized[candidate]
                    break
        if column is None:
            raise ValueError(f"Could not find an f0 column in {path}")

        values = []
        for row in reader:
            try:
                values.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                values.append(0.0)

    return torch.tensor(values, dtype=torch.float32).clamp_min(0.0)


def _as_batch_f0(
    f0_hz: torch.Tensor,
    batch_size: int,
    n_frames: int,
    device: torch.device,
) -> torch.Tensor:
    f0_hz = torch.as_tensor(f0_hz, dtype=torch.float32, device=device)
    if f0_hz.dim() == 1:
        f0_hz = f0_hz.unsqueeze(0)
    if f0_hz.dim() != 2:
        raise ValueError("f0_hz must have shape (frames,) or (batch, frames).")

    if f0_hz.shape[-1] != n_frames:
        f0_hz = F.interpolate(
            f0_hz.unsqueeze(1),
            size=n_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    if f0_hz.shape[0] == 1 and batch_size > 1:
        return f0_hz.expand(batch_size, -1)
    if f0_hz.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} f0 contour(s), got {f0_hz.shape[0]}.")
    return f0_hz


def harmonic_mask(
    f0_hz: torch.Tensor,
    batch_size: int,
    n_freq_bins: int,
    n_frames: int,
    sample_rate: int,
    n_fft: int,
    n_harmonics: int = 32,
    width_bins: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a binary mask that keeps f0 and its harmonics."""
    device = device or torch.device("cpu")
    f0_hz = _as_batch_f0(f0_hz, batch_size, n_frames, device)
    width_bins = max(0, int(width_bins))
    n_harmonics = max(1, int(n_harmonics))

    harmonics = torch.arange(1, n_harmonics + 1, device=device).view(1, -1, 1)
    harmonic_hz = f0_hz.unsqueeze(1) * harmonics
    valid = (harmonic_hz > 0.0) & (harmonic_hz <= sample_rate * 0.5)

    bin_hz = sample_rate / n_fft
    harmonic_bins = torch.round(harmonic_hz / bin_hz).long().clamp(0, n_freq_bins - 1)

    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(harmonic_bins)
    frame_idx = torch.arange(n_frames, device=device).view(1, 1, -1).expand_as(harmonic_bins)

    mask = torch.zeros(batch_size, n_freq_bins, n_frames, device=device)
    for offset in range(-width_bins, width_bins + 1):
        bins = (harmonic_bins + offset).clamp(0, n_freq_bins - 1)
        mask[batch_idx[valid], bins[valid], frame_idx[valid]] = 1.0
    return mask


def _dual_window(window: torch.Tensor, hop_length: int, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.zeros_like(window)
    for start in range(min(hop_length, window.numel())):
        idx = torch.arange(start, window.numel(), hop_length, device=window.device)
        denom[idx] = torch.sum(window[idx] ** 2)
    return window / denom.clamp_min(eps)


def reproducing_kernel(
    n_fft: int,
    win_length: int,
    hop_length: int,
    kernel_floor: float = 1e-3,
) -> torch.Tensor:
    """Approximate the STFT reproducing kernel used to smooth sparse masks."""
    key = (int(n_fft), int(win_length), int(hop_length), float(kernel_floor))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    window = torch.hann_window(win_length)
    dual = _dual_window(window, hop_length)
    kernel = torch.stft(
        dual,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
        onesided=False,
    )
    kernel = torch.abs(torch.fft.fftshift(kernel, dim=0))
    kernel = kernel / kernel.max().clamp_min(1e-8)

    active = kernel >= kernel_floor
    if bool(active.any()):
        rows, cols = active.nonzero(as_tuple=True)
        kernel = kernel[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        kernel = torch.where(kernel >= kernel_floor, kernel, torch.zeros_like(kernel))
    else:
        kernel = torch.ones(1, 1)

    if kernel.shape[0] % 2 == 0:
        kernel = F.pad(kernel, (0, 0, 0, 1))
    if kernel.shape[1] % 2 == 0:
        kernel = F.pad(kernel, (0, 1, 0, 0))

    kernel = kernel / kernel.sum().clamp_min(1e-8)
    _KERNEL_CACHE[key] = kernel
    return kernel


def smooth_mask(
    mask: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    kernel_floor: float = 1e-3,
) -> torch.Tensor:
    kernel = reproducing_kernel(n_fft, win_length, hop_length, kernel_floor)
    kernel = kernel.to(device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(0)
    smoothed = F.conv2d(
        mask.unsqueeze(1),
        kernel,
        padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2),
    ).squeeze(1)
    return smoothed.clamp(0.0, 1.0)


def masked_spectrogram(
    waveform: torch.Tensor,
    f0_hz: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 8192,
    win_length: int | None = None,
    hop_length: int | None = None,
    power: float = 1.0,
    normalized: bool = False,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    n_harmonics: int = 32,
    width_bins: int = 1,
    kernel_floor: float = 1e-3,
    return_mask: bool = False,
):
    """Compute an STFT magnitude after keeping smoothed f0 harmonics."""
    if torch.is_tensor(waveform):
        waveform = waveform.to(dtype=torch.float32)
    else:
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError("waveform must have shape (time,) or (batch, time).")

    win_length = int(win_length or n_fft)
    hop_length = int(hop_length or n_fft // 32)
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=normalized,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
        return_complex=True,
    )

    sparse_mask = harmonic_mask(
        f0_hz=f0_hz,
        batch_size=spec.shape[0],
        n_freq_bins=spec.shape[-2],
        n_frames=spec.shape[-1],
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_harmonics=n_harmonics,
        width_bins=width_bins,
        device=waveform.device,
    )
    mask = smooth_mask(
        sparse_mask,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        kernel_floor=kernel_floor,
    )
    masked = torch.abs(spec * mask.to(spec.dtype))
    if power != 1.0:
        masked = masked ** float(power)

    if return_mask:
        return masked, mask
    return masked
