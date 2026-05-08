import numpy as np
import torch
import torch.nn.functional as F

import csv
import math
import soundfile as sf
from pathlib import Path
from typing import Optional, Union

_REPRODUCING_KERNEL_CACHE: dict[tuple[int, int, int, int, int, float], torch.Tensor] = {}

def _align_f0_to_frames(f0: torch.Tensor, n_frames: int) -> torch.Tensor:
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
    if f0.shape[-1] == n_frames:
        return f0
    return F.interpolate(
        f0.unsqueeze(1),
        size=n_frames,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def _build_harmonic_sparse_mask(
    f0_frames: torch.Tensor,
    n_freq_bins: int,
    sample_rate: int,
    n_fft: int,
    max_harmonics: int,
    bin_radius: int = 1,
) -> torch.Tensor:
    mask = torch.zeros(
        n_freq_bins,
        f0_frames.shape[-1],
        device=f0_frames.device,
        dtype=torch.float32,
    )
    if max_harmonics <= 0:
        return mask

    nyquist = sample_rate * 0.5
    freq_res = sample_rate / max(n_fft, 1)
    harmonic_idx = torch.arange(
        1,
        max_harmonics + 1,
        device=f0_frames.device,
        dtype=f0_frames.dtype,
    ).view(-1, 1)
    harmonic_freqs = harmonic_idx * f0_frames.view(1, -1)
    valid = (harmonic_freqs > 0.0) & (harmonic_freqs <= nyquist)
    if not bool(valid.any()):
        return mask

    harmonic_bins = torch.round(harmonic_freqs / max(freq_res, 1e-8)).long()
    harmonic_bins = torch.clamp(harmonic_bins, 0, n_freq_bins - 1)
    frame_idx = torch.arange(f0_frames.shape[-1], device=f0_frames.device).view(1, -1)
    frame_idx = frame_idx.expand_as(harmonic_bins)

    bin_radius = max(0, int(bin_radius))
    if bin_radius == 0:
        mask[harmonic_bins[valid], frame_idx[valid]] = 1.0
        return mask

    offsets = torch.arange(
        -bin_radius,
        bin_radius + 1,
        device=f0_frames.device,
        dtype=harmonic_bins.dtype,
    ).view(-1, 1, 1)
    expanded_bins = torch.clamp(harmonic_bins.unsqueeze(0) + offsets, 0, n_freq_bins - 1)
    expanded_frames = frame_idx.unsqueeze(0).expand_as(expanded_bins)
    expanded_valid = valid.unsqueeze(0).expand_as(expanded_bins)
    mask[expanded_bins[expanded_valid], expanded_frames[expanded_valid]] = 1.0
    return mask


def _approximate_dual_window(
    analysis_window: torch.Tensor,
    hop_length: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Approximate the canonical dual window from the frame-operator diagonal:

        w_dual[n] ~= w[n] / sum_m |w[n - m * hop]|^2
    """
    win_length = int(analysis_window.numel())
    hop_length = max(1, int(hop_length))

    n = torch.arange(win_length, device=analysis_window.device)
    denom = torch.zeros_like(analysis_window)
    max_shift = math.ceil(max(1, win_length - 1) / hop_length) + 1

    for m in range(-max_shift, max_shift + 1):
        shifted_idx = n - (m * hop_length)
        valid = (shifted_idx >= 0) & (shifted_idx < win_length)
        if bool(valid.any()):
            w_vals = analysis_window[shifted_idx[valid].long()]
            denom[valid] += torch.abs(w_vals) ** 2

    denom = torch.clamp(denom, min=eps)
    return analysis_window / denom


def _get_reproducing_kernel(
    n_fft: int,
    hop_length: int,
    win_length: int,
    time_radius: Optional[int],
    freq_radius: Optional[int],
    kernel_floor: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate |<w_dual, w_{m,n}>| for the current STFT setup and cache it.

    Approximate the reproducing kernel with the magnitude STFT of the approximate
    dual window under analysis window w: |V_w w_dual|.

    If `time_radius` or `freq_radius` is None, support is selected
    automatically from entries above `kernel_floor` in the normalized kernel.
    """
    time_radius_key = -1 if time_radius is None else int(time_radius)
    freq_radius_key = -1 if freq_radius is None else int(freq_radius)
    key = (
        int(n_fft),
        int(hop_length),
        int(win_length),
        time_radius_key,
        freq_radius_key,
        float(kernel_floor),
    )
    if key in _REPRODUCING_KERNEL_CACHE:
        return _REPRODUCING_KERNEL_CACHE[key]

    device = torch.device("cpu")
    n_fft = max(2, int(n_fft))
    hop_length = max(1, int(hop_length))
    win_length = max(1, int(win_length))
    if time_radius is None:
        time_radius_eff = None
    else:
        time_radius_eff = max(0, int(time_radius))
    if freq_radius is None:
        freq_radius_eff = None
    else:
        freq_radius_eff = min(max(0, int(freq_radius)), n_fft // 2)
    kernel_floor = max(0.0, float(kernel_floor))

    analysis_window = torch.hann_window(win_length, device=device, dtype=torch.float32)
    dual_window = _approximate_dual_window(
        analysis_window=analysis_window,
        hop_length=hop_length,
        eps=eps,
    )

    # Use zero-padding (not reflection) to estimate frame shifts around n=0.
    # Reflection introduces artificial mirrored content and distorts kernel shape.
    window_stft = torch.stft(
        dual_window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=analysis_window,
        center=True,
        pad_mode="constant",
        return_complex=True,
        onesided=False,
    )
    window_stft_mag = torch.abs(torch.fft.fftshift(window_stft, dim=0))
    window_norm = window_stft_mag / torch.clamp(window_stft_mag.max(), min=eps)

    center_f = window_stft_mag.shape[0] // 2
    center_t = window_stft_mag.shape[1] // 2
    if freq_radius_eff is None or time_radius_eff is None:
        # If kernel_floor is <= 0, keep auto-support stable by using a tiny
        # positive support floor. Otherwise the support expands to the full
        # FFT plane, which over-smooths masks and is very expensive.
        support_floor = kernel_floor if kernel_floor > 0.0 else 1e-6
        active = window_norm >= support_floor
        if bool(active.any()):
            active_idx = active.nonzero(as_tuple=False)
            auto_freq_radius = int(torch.max(torch.abs(active_idx[:, 0] - center_f)).item())
            auto_time_radius = int(torch.max(torch.abs(active_idx[:, 1] - center_t)).item())
        else:
            auto_freq_radius = 0
            auto_time_radius = 0
        if freq_radius_eff is None:
            freq_radius_eff = auto_freq_radius
        if time_radius_eff is None:
            time_radius_eff = auto_time_radius

    # Keep radius strictly inside available fftshift support for even n_fft.
    max_freq_radius = min(center_f, window_stft_mag.shape[0] - 1 - center_f)
    freq_radius_eff = min(max(0, int(freq_radius_eff)), max_freq_radius)
    time_radius_eff = max(0, int(time_radius_eff))

    f_lo = max(0, center_f - freq_radius_eff)
    f_hi = min(window_stft_mag.shape[0], center_f + freq_radius_eff + 1)
    t_lo = max(0, center_t - time_radius_eff)
    t_hi = min(window_stft_mag.shape[1], center_t + time_radius_eff + 1)
    kernel = window_stft_mag[f_lo:f_hi, t_lo:t_hi]

    target_f = 2 * freq_radius_eff + 1
    target_t = 2 * time_radius_eff + 1
    if kernel.shape[0] < target_f or kernel.shape[1] < target_t:
        pad_f = max(0, target_f - kernel.shape[0])
        pad_t = max(0, target_t - kernel.shape[1])
        pad_top = pad_f // 2
        pad_bottom = pad_f - pad_top
        pad_left = pad_t // 2
        pad_right = pad_t - pad_left
        kernel = F.pad(kernel, (pad_left, pad_right, pad_top, pad_bottom))
    if kernel.shape[0] > target_f:
        extra = kernel.shape[0] - target_f
        start = extra // 2
        kernel = kernel[start : start + target_f, :]
    if kernel.shape[1] > target_t:
        extra = kernel.shape[1] - target_t
        start = extra // 2
        kernel = kernel[:, start : start + target_t]
    if kernel.numel() == 0:
        kernel = torch.ones(1, 1, device=device, dtype=torch.float32)

    kernel = kernel / torch.clamp(kernel.max(), min=eps)
    kernel = torch.where(kernel >= kernel_floor, kernel, torch.zeros_like(kernel))
    if float(torch.sum(kernel)) <= 0.0:
        kernel = torch.ones_like(kernel)
    kernel = kernel / torch.clamp(torch.sum(kernel), min=eps)

    _REPRODUCING_KERNEL_CACHE[key] = kernel
    return kernel


def _resolve_f0_csv_path(
    wav_path: Union[str, Path],
    f0_contour_path: Optional[Union[str, Path]] = None,
) -> Path:
    if f0_contour_path is not None:
        f0_path = Path(f0_contour_path).expanduser().resolve()
        if not f0_path.exists():
            raise FileNotFoundError(f"F0 contour CSV not found: {f0_path}")
        return f0_path

    wav_path = Path(wav_path).expanduser().resolve()
    candidates = [
        wav_path.parent / "f0_corrected" / f"{wav_path.stem}.f0.csv",
        wav_path.parent / f"{wav_path.stem}.f0.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find F0 CSV for {wav_path.name}. Expected "
        f"{candidates[0].name} in {candidates[0].parent}. "
        "Please run extract_f0 and annotate_f0 first."
    )


def _load_f0_contour_from_csv(
    f0_csv_path: Union[str, Path],
    assume_frequency_is_f1: Optional[bool] = None,
) -> torch.Tensor:
    f0_csv_path = Path(f0_csv_path).expanduser().resolve()
    if not f0_csv_path.exists():
        raise FileNotFoundError(f"F0 CSV not found: {f0_csv_path}")

    with open(f0_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Invalid CSV without header: {f0_csv_path}")

        key_map = {}
        for key in reader.fieldnames:
            if key is not None:
                key_map[key.strip().lower()] = key

        freq_key = None
        for candidate in ("f0_hz", "f0", "frequency"):
            if candidate in key_map:
                freq_key = key_map[candidate]
                break
        if freq_key is None:
            raise ValueError(
                f"Could not find an F0 column in {f0_csv_path}. "
                "Expected one of: f0_hz, f0, frequency."
            )

        values = []
        for row in reader:
            raw = row.get(freq_key, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            values.append(value)

    if len(values) == 0:
        raise ValueError(f"F0 CSV has no rows: {f0_csv_path}")

    f0 = np.asarray(values, dtype=np.float32)
    freq_key_l = freq_key.strip().lower()
    if freq_key_l == "frequency":
        if assume_frequency_is_f1 is None:
            assume_frequency_is_f1 = f0_csv_path.parent.name == "f0_corrected"
        if assume_frequency_is_f1:
            f0 = f0 * 0.5

    f0[~np.isfinite(f0)] = 0.0
    f0 = np.maximum(f0, 0.0)
    return torch.from_numpy(f0)


def _build_smoothed_harmonic_mask(
    f0_contour: torch.Tensor,
    batch_size: int,
    n_frames: int,
    n_freq_bins: int,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_harmonics: int,
    width_bins: int,
    kernel_time_radius: Optional[int],
    kernel_freq_radius: Optional[int],
    kernel_floor: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if f0_contour.dim() == 1:
        f0_contour = f0_contour.unsqueeze(0)
    elif f0_contour.dim() != 2:
        raise ValueError("f0_contour must be 1D or 2D (batch, frames).")
    f0_contour = f0_contour.to(dtype=torch.float32)
    f0_aligned = _align_f0_to_frames(f0_contour, n_frames=n_frames)
    if f0_aligned.shape[0] == 1 and batch_size > 1:
        f0_aligned = f0_aligned.expand(batch_size, -1)
    elif f0_aligned.shape[0] != batch_size:
        f0_aligned = f0_aligned[:batch_size]

    mask_sparse = torch.zeros(
        batch_size,
        n_freq_bins,
        n_frames,
        device=f0_aligned.device,
        dtype=torch.float32,
    )
    for b_idx in range(batch_size):
        mask_sparse[b_idx] = _build_harmonic_sparse_mask(
            f0_aligned[b_idx],
            n_freq_bins=n_freq_bins,
            sample_rate=sample_rate,
            n_fft=n_fft,
            max_harmonics=max(1, int(n_harmonics)),
            bin_radius=max(0, int(width_bins)),
        )

    kernel = _get_reproducing_kernel(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        time_radius=kernel_time_radius,
        freq_radius=kernel_freq_radius,
        kernel_floor=float(kernel_floor),
        eps=eps,
    ).to(dtype=torch.float32, device=mask_sparse.device)
    kernel = torch.flip(kernel, dims=(0, 1)).unsqueeze(0).unsqueeze(0)

    mask_smooth = F.conv2d(
        mask_sparse.unsqueeze(1),
        kernel,
        padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2),
    ).squeeze(1)
    mask_smooth = torch.clamp(mask_smooth, 0.0, 1.0)

    # Eq. 16 smoothing is always applied; no post-threshold rebinarization.
    mask_used = mask_smooth

    return mask_sparse, mask_smooth, mask_used


# def masked_elespectrogram(
#     waveform: torch.Tensor,
#     f0_contour: Optional[Union[torch.Tensor, np.ndarray]] = None,
#     f0_contour_path: Optional[Union[str, Path]] = None,
#     wav_path: Optional[Union[str, Path]] = None,
#     width_bins: int = 1,
#     n_harmonics: int = 32,
#     sample_rate: int = 16000,
#     n_fft: int = 8192,
#     win_length: Optional[int] = None,
#     hop_length: Optional[int] = None,
#     fmin: float = 0.0,
#     fmax: Optional[float] = None,
#     n_mels: int = 128,
#     power: float = 1.0,
#     normalized: bool = False,
#     center: bool = True,
#     pad_mode: str = "reflect",
#     onesided: bool = True,
#     norm: Optional[str] = None,
#     scale: str = "elelog",
#     kernel_time_radius: Optional[int] = None,
#     kernel_freq_radius: Optional[int] = None,
#     kernel_floor: float = 1e-3,
#     assume_frequency_is_f1: Optional[bool] = None,
#     return_details: bool = False,
# ):
#     transform = MaskedEleSpectrogram(
#         sample_rate=sample_rate,
#         n_fft=n_fft,
#         win_length=win_length,
#         hop_length=hop_length,
#         fmin=fmin,
#         fmax=fmax,
#         n_mels=n_mels,
#         power=power,
#         normalized=normalized,
#         center=center,
#         pad_mode=pad_mode,
#         onesided=onesided,
#         norm=norm,
#         scale=scale,
#     ).to(waveform.device if torch.is_tensor(waveform) else torch.device("cpu"))
#     return transform(
#         waveform=waveform,
#         f0_contour=f0_contour,
#         f0_contour_path=f0_contour_path,
#         wav_path=wav_path,
#         width_bins=width_bins,
#         n_harmonics=n_harmonics,
#         kernel_time_radius=kernel_time_radius,
#         kernel_freq_radius=kernel_freq_radius,
#         kernel_floor=kernel_floor,
#         assume_frequency_is_f1=assume_frequency_is_f1,
#         return_details=return_details,
#     )


def harmonic_enhancement(
    wav_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    f0_contour_path: Optional[Union[str, Path]] = None,
    width_bins: int = 1,
    n_harmonics: int = 32,
    sample_rate: int = 16000,
    n_fft: int = 8192,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    normalized: bool = False,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    kernel_time_radius: Optional[int] = None,
    kernel_freq_radius: Optional[int] = None,
    kernel_floor: float = 1e-3,
    assume_frequency_is_f1: Optional[bool] = None,
    return_details: bool = False,
):
    """
    Harmonic masking enhancement on a WAV file:
    STFT -> harmonic mask -> ISTFT -> cleaned waveform.
    """
    wav_path = Path(wav_path).expanduser().resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    f0_csv_path = _resolve_f0_csv_path(wav_path, f0_contour_path=f0_contour_path)

    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    if sr != sample_rate:
        import librosa

        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    waveform = torch.from_numpy(audio_np.astype(np.float32))
    _, details = masked_spectrogram(
        waveform=waveform,
        f0_contour_path=f0_csv_path,
        width_bins=width_bins,
        n_harmonics=n_harmonics,
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        normalized=normalized,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
        kernel_time_radius=kernel_time_radius,
        kernel_freq_radius=kernel_freq_radius,
        kernel_floor=kernel_floor,
        assume_frequency_is_f1=assume_frequency_is_f1,
        return_complex=True,
        return_details=True,
    )

    masked_complex = details["masked_complex_spectrogram"].to(torch.complex64)
    n_fft = int(details["n_fft"])
    win_length = int(details["win_length"])
    hop_length = int(details["hop_length"])
    window = torch.hann_window(win_length, dtype=torch.float32)

    enhanced = torch.istft(
        masked_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=waveform.shape[-1],
        center=center,
    )
    enhanced_np = enhanced.detach().cpu().numpy().astype(np.float32)

    if output_path is None:
        output_path = wav_path.with_name(f"{wav_path.stem}_harmonic_enhanced.wav")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), enhanced_np, sr)

    result = {
        "audio": enhanced,
        "sample_rate": sr,
        "output_path": str(output_path),
        "f0_contour_path": str(f0_csv_path),
    }
    if return_details:
        result["details"] = details
    return result

def masked_spectrogram(
    waveform: torch.Tensor,
    f0_contour: Optional[Union[torch.Tensor, np.ndarray]] = None,
    f0_contour_path: Optional[Union[str, Path]] = None,
    wav_path: Optional[Union[str, Path]] = None,
    width_bins: int = 1,
    n_harmonics: int = 32,
    sample_rate: int = 16000,
    n_fft: int = 8192,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    power: float = 1.0,
    normalized: bool = False,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    kernel_time_radius: Optional[int] = None,
    kernel_freq_radius: Optional[int] = None,
    kernel_floor: float = 1e-3,
    assume_frequency_is_f1: Optional[bool] = None,
    return_complex: bool = False,
    return_details: bool = False,
):
    """
    Compute a harmonic-mask enhanced STFT representation.

    The sparse mask is built at F0 + harmonics with +/- `width_bins` support and
    always smoothed using Eq. 16 style reproducing-kernel convolution.
    Kernel support is automatic by default (`kernel_*_radius=None`).
    """
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        squeeze_output = True
    elif waveform.dim() == 2:
        squeeze_output = False
    else:
        raise ValueError("waveform must be 1D or 2D (batch, time).")

    waveform = waveform.to(dtype=torch.float32)
    if win_length is None:
        win_length = int(n_fft)
    if hop_length is None:
        hop_length = max(1, int(n_fft) // 32)
    n_fft = int(n_fft)
    win_length = int(win_length)
    hop_length = int(hop_length)

    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    spec_complex = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=True,
    )

    n_freq_bins = spec_complex.shape[-2]
    n_frames = spec_complex.shape[-1]
    if f0_contour is not None:
        if not torch.is_tensor(f0_contour):
            f0_values = torch.as_tensor(f0_contour, dtype=torch.float32)
        else:
            f0_values = f0_contour.to(dtype=torch.float32)
        f0_values = f0_values.to(device=waveform.device)
    else:
        if f0_contour_path is None:
            if wav_path is None:
                raise ValueError("Provide one of: f0_contour, f0_contour_path, wav_path.")
            f0_contour_path = _resolve_f0_csv_path(wav_path)

        f0_values = _load_f0_contour_from_csv(
            f0_contour_path,
            assume_frequency_is_f1=assume_frequency_is_f1,
        ).to(device=waveform.device)

    mask_sparse, mask_smooth, mask_used = _build_smoothed_harmonic_mask(
        f0_contour=f0_values,
        batch_size=spec_complex.shape[0],
        n_frames=n_frames,
        n_freq_bins=n_freq_bins,
        sample_rate=int(sample_rate),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_harmonics=n_harmonics,
        width_bins=width_bins,
        kernel_time_radius=kernel_time_radius,
        kernel_freq_radius=kernel_freq_radius,
        kernel_floor=kernel_floor,
    )
    masked_complex = spec_complex * mask_used.to(dtype=spec_complex.dtype)

    if return_complex:
        output_spec = masked_complex
    else:
        output_spec = torch.abs(masked_complex)
        if power != 1.0:
            output_spec = output_spec ** float(power)

    if squeeze_output:
        output_spec = output_spec.squeeze(0)
        masked_complex = masked_complex.squeeze(0)

    if not return_details:
        return output_spec

    details = {
        "f0_contour": f0_values,
        "mask_sparse": mask_sparse,
        "mask_smooth": mask_smooth,
        "mask_used": mask_used,
        "spectrogram_complex": spec_complex.squeeze(0) if squeeze_output else spec_complex,
        "masked_complex_spectrogram": masked_complex,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "f0_contour_path": str(Path(f0_contour_path).expanduser()) if f0_contour_path is not None else None,
    }
    return output_spec, details
