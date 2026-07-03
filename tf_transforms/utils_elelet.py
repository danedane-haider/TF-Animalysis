import math

import numpy as np
import torch
import torch.nn.functional as F

from .utils_auditory_scales import audspace_mod, freqtoaud_mod, freqtoaud

####################################################################################################
#####################         ELELET Routines                      #################################
####################################################################################################

FFT_BACKENDS = ("fft", "fft_decimated")


def _validate_fft_backend(backend: str) -> None:
    if backend not in FFT_BACKENDS:
        choices = ", ".join(repr(choice) for choice in FFT_BACKENDS)
        raise ValueError(f"backend must be one of {choices}; got {backend!r}.")


def _numpy_kernel_fft(kernels: np.ndarray, length: int) -> np.ndarray:
    """Return centered, length-``length`` kernel spectra."""
    kernel_size = kernels.shape[-1]
    kernels_long = np.pad(
        kernels,
        ((0, 0), (0, length - kernel_size)),
        mode="constant",
        constant_values=0,
    )
    kernels_centered = np.roll(
        kernels_long,
        shift=-kernel_size // 2,
        axis=-1,
    )
    return np.fft.fft(kernels_centered, n=length, axis=-1)


def _torch_kernel_fft(
    kernels: torch.Tensor,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return centered, length-``length`` kernel spectra."""
    if kernels.dim() == 3 and kernels.shape[-2] == 1:
        kernels = kernels.squeeze(-2)
    if kernels.dim() != 2:
        raise ValueError("kernels must have shape (channels, kernel_length).")

    kernel_size = kernels.shape[-1]
    kernels = kernels.to(device)
    kernels_long = F.pad(
        kernels,
        (0, length - kernel_size),
        mode="constant",
        value=0,
    )
    kernels_centered = torch.roll(
        kernels_long,
        shifts=-kernel_size // 2,
        dims=-1,
    )
    return torch.fft.fft(kernels_centered, n=length, dim=-1)


def _numpy_decimated_ifft(
    spectrum: np.ndarray,
    stride: int,
    start: int,
    output_length: int,
) -> np.ndarray:
    """Evaluate an inverse DFT only at ``start + m * stride``.

    The aliasing identity reduces the inverse-transform length by
    ``gcd(transform_length, stride)``. For fixed-size ML inputs whose length is
    a multiple of the stride, this is the full stride factor.
    """
    length = spectrum.shape[-1]
    reduction = math.gcd(length, stride)
    if reduction == 1:
        samples = np.fft.ifft(spectrum, axis=-1)
        indices = start + np.arange(output_length) * stride
        return np.take(samples, indices, axis=-1)

    reduced_length = length // reduction
    if start:
        bins = np.arange(length)
        phase = np.exp(2j * np.pi * bins * start / length)
        spectrum = spectrum * phase

    aliased = spectrum.reshape(*spectrum.shape[:-1], reduction, reduced_length)
    aliased = aliased.sum(axis=-2) / reduction
    samples = np.fft.ifft(aliased, axis=-1)

    reduced_stride = stride // reduction
    indices = (np.arange(output_length) * reduced_stride) % reduced_length
    return np.take(samples, indices, axis=-1)


def _torch_decimated_ifft(
    spectrum: torch.Tensor,
    stride: int,
    start: int,
    output_length: int,
) -> torch.Tensor:
    """Torch equivalent of :func:`_numpy_decimated_ifft`."""
    length = spectrum.shape[-1]
    reduction = math.gcd(length, stride)
    if reduction == 1:
        samples = torch.fft.ifft(spectrum, dim=-1)
        indices = start + torch.arange(
            output_length,
            device=spectrum.device,
        ) * stride
        return torch.index_select(samples, -1, indices)

    reduced_length = length // reduction
    if start:
        real_dtype = spectrum.real.dtype
        bins = torch.arange(length, device=spectrum.device, dtype=real_dtype)
        phase = torch.exp((2j * torch.pi * start / length) * bins)
        spectrum = spectrum * phase

    aliased = spectrum.reshape(
        *spectrum.shape[:-1],
        reduction,
        reduced_length,
    )
    aliased = aliased.sum(dim=-2) / reduction
    samples = torch.fft.ifft(aliased, dim=-1)

    reduced_stride = stride // reduction
    indices = (
        torch.arange(output_length, device=spectrum.device) * reduced_stride
    ) % reduced_length
    return torch.index_select(samples, -1, indices)


def circ_conv_numpy(
    x: np.ndarray,
    kernels: np.ndarray,
    d: int = 1,
    pad_mode: str = "circular",
    backend: str = "fft",
    kernel_fft: np.ndarray | None = None,
    channel_batch_size: int | None = None,
) -> np.ndarray:
    """FFT convolution with optional exact stride-aware inverse transforms.

    ``backend="fft"`` computes every convolved sample before striding.
    ``backend="fft_decimated"`` uses frequency-domain aliasing to evaluate
    only the requested samples. The latter is especially effective when the
    transform length is divisible by ``d``.
    """
    _validate_fft_backend(backend)
    if d <= 0:
        raise ValueError("d must be a positive integer.")
    if channel_batch_size is not None and channel_batch_size <= 0:
        raise ValueError("channel_batch_size must be positive or None.")

    x = np.asarray(x)
    kernels = np.asarray(kernels)
    squeeze_batch = x.ndim == 1
    if squeeze_batch:
        x = x[np.newaxis, :]
    elif x.ndim == 3 and x.shape[-2] == 1:
        x = np.squeeze(x, axis=-2)
    if x.ndim != 2:
        raise ValueError(
            "x must have shape (time,), (batch, time), or (batch, 1, time)."
        )
    if kernels.ndim == 3 and kernels.shape[-2] == 1:
        kernels = np.squeeze(kernels, axis=-2)
    if kernels.ndim != 2:
        raise ValueError("kernels must have shape (channels, kernel_length).")

    original_length = x.shape[-1]
    kernel_size = kernels.shape[-1]

    crop_start = 0
    if pad_mode != "circular":
        pad = kernel_size // 2
        if pad_mode == "reflect" and original_length <= pad:
            raise ValueError(
                "reflect padding requires audio to be longer than half the kernel size."
            )
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode=pad_mode)
        crop_start = pad

    L = x.shape[-1]
    if L < kernel_size:
        x = np.pad(
            x,
            [(0, 0)] * (x.ndim - 1) + [(0, kernel_size - L)],
            mode="constant",
        )
        L = x.shape[-1]

    x_fft = np.fft.fft(x, n=L, axis=-1)
    if kernel_fft is None:
        kernel_fft = _numpy_kernel_fft(kernels, L)
    elif kernel_fft.shape != (kernels.shape[0], L):
        raise ValueError(
            "kernel_fft must have shape "
            f"({kernels.shape[0]}, {L}); got {kernel_fft.shape}."
        )

    num_channels = kernels.shape[0]
    batch_size = channel_batch_size or num_channels
    if pad_mode == "circular":
        output_length = (L + d - 1) // d
        output_start = 0
    else:
        output_length = (original_length + d - 1) // d
        output_start = crop_start

    if backend == "fft_decimated" and output_start:
        bins = np.arange(L)
        phase = np.exp(2j * np.pi * bins * output_start / L)
        x_fft = x_fft * phase
        output_start = 0

    outputs = []
    for channel_start in range(0, num_channels, batch_size):
        channel_stop = min(channel_start + batch_size, num_channels)
        y_fft = (
            x_fft[..., np.newaxis, :]
            * kernel_fft[channel_start:channel_stop]
        )

        if backend == "fft_decimated":
            output = _numpy_decimated_ifft(
                y_fft,
                stride=d,
                start=output_start,
                output_length=output_length,
            )
        else:
            output = np.fft.ifft(y_fft, axis=-1)
            if pad_mode != "circular":
                output = output[..., crop_start : crop_start + original_length]
            output = output[..., ::d]
        outputs.append(output)

    result = np.concatenate(outputs, axis=-2)
    return result[0] if squeeze_batch else result


def upsample(x: torch.Tensor, d: int) -> torch.Tensor:
    N = x.shape[-1] * d
    x_up = F.pad(torch.zeros_like(x), (0, N - x.shape[-1]))
    x_up[:, :, ::d] = x
    return x_up

def circ_conv(
    x: torch.Tensor,
    kernels: torch.Tensor,
    d: int = 1,
    pad_mode: str = "circular",
    backend: str = "fft",
    kernel_fft: torch.Tensor | None = None,
    channel_batch_size: int | None = None,
) -> torch.Tensor:
    """FFT convolution with optional downsampling.

    By default this uses circular boundary conditions. Set ``pad_mode="reflect"``
    to mirror-pad the signal before convolution, similar to ``torch.stft`` with
    ``center=True, pad_mode="reflect"``.

    Args:
        x (torch.Tensor): Input signal of shape (..., signal_length)
        kernels (torch.Tensor): Filter kernels of shape (num_channels, 1, kernel_length)
            or (num_channels, kernel_length)
        d (int): Downsampling factor (stride). Default: 1
        pad_mode (str): ``"circular"`` for the old wraparound behavior, or a
            ``torch.nn.functional.pad`` mode such as ``"reflect"``.
        backend (str): ``"fft"`` for the reference full inverse FFT or
            ``"fft_decimated"`` to evaluate only strided output samples.
        kernel_fft (torch.Tensor, optional): Precomputed centered kernel spectra.
        channel_batch_size (int, optional): Limit channels processed together.

    Returns:
        torch.Tensor: Convolved and downsampled output of shape (..., num_channels, output_length)

    Note:
        Kernels are automatically zero-padded and centered.

    Example:
        >>> x = torch.randn(1, 1000)
        >>> kernels = torch.randn(40, 128)
        >>> y = circ_conv(x, kernels, d=4)
    """
    _validate_fft_backend(backend)
    if d <= 0:
        raise ValueError("d must be a positive integer.")
    if channel_batch_size is not None and channel_batch_size <= 0:
        raise ValueError("channel_batch_size must be positive or None.")

    squeeze_batch = x.dim() == 1
    if squeeze_batch:
        x = x.unsqueeze(0)
    elif x.dim() == 3 and x.shape[-2] == 1:
        x = x.squeeze(-2)
    if x.dim() != 2:
        raise ValueError(
            "x must have shape (time,), (batch, time), or (batch, 1, time)."
        )
    if kernels.dim() == 3 and kernels.shape[-2] == 1:
        kernels = kernels.squeeze(-2)
    if kernels.dim() != 2:
        raise ValueError("kernels must have shape (channels, kernel_length).")

    original_length = x.shape[-1]
    kernel_size = kernels.shape[-1]
    crop_start = 0

    if pad_mode != "circular":
        pad = kernel_size // 2
        if pad_mode == "reflect" and original_length <= pad:
            raise ValueError(
                "reflect padding requires audio to be longer than half the kernel size."
            )
        x = F.pad(x, (pad, pad), mode=pad_mode)
        crop_start = pad

    L = x.shape[-1]
    if L < kernel_size:
        x = F.pad(x, (0, kernel_size - L), mode="constant", value=0)
        L = x.shape[-1]

    x = x.to(kernels.dtype)
    x_fft = torch.fft.fft(x, n=L, dim=-1)
    if kernel_fft is None:
        kernel_fft = _torch_kernel_fft(kernels, L, x.device)
    elif kernel_fft.shape != (kernels.shape[0], L):
        raise ValueError(
            "kernel_fft must have shape "
            f"({kernels.shape[0]}, {L}); got {tuple(kernel_fft.shape)}."
        )
    else:
        kernel_fft = kernel_fft.to(x.device)

    num_channels = kernels.shape[0]
    batch_size = channel_batch_size or num_channels
    if pad_mode == "circular":
        output_length = (L + d - 1) // d
        output_start = 0
    else:
        output_length = (original_length + d - 1) // d
        output_start = crop_start

    if backend == "fft_decimated" and output_start:
        real_dtype = x_fft.real.dtype
        bins = torch.arange(L, device=x_fft.device, dtype=real_dtype)
        phase = torch.exp((2j * torch.pi * output_start / L) * bins)
        x_fft = x_fft * phase
        output_start = 0

    outputs = []
    for channel_start in range(0, num_channels, batch_size):
        channel_stop = min(channel_start + batch_size, num_channels)
        y_fft = x_fft.unsqueeze(-2) * kernel_fft[channel_start:channel_stop]

        if backend == "fft_decimated":
            output = _torch_decimated_ifft(
                y_fft,
                stride=d,
                start=output_start,
                output_length=output_length,
            )
        else:
            output = torch.fft.ifft(y_fft, dim=-1)
            if pad_mode != "circular":
                output = output[..., crop_start : crop_start + original_length]
            output = output[..., ::d]
        outputs.append(output)

    result = torch.cat(outputs, dim=-2)
    return result[0] if squeeze_batch else result


def circ_conv_transpose(
    y: torch.Tensor, kernels: torch.Tensor, d: int = 1
) -> torch.Tensor:
    """Transpose (adjoint) of circular convolution with upsampling.
    Used in synthesis/decoder operations of filterbanks.

    Args:
        y (torch.Tensor): Input coefficients of shape (..., num_channels, num_frames)
        kernels (torch.Tensor): Filter kernels of shape (num_channels, 1, kernel_length)
            or (num_channels, kernel_length)
        d (int): Upsampling factor (stride). Default: 1

    Returns:
        torch.Tensor: Reconstructed signal of shape (..., 1, signal_length)

    Note:
        This is the mathematical adjoint, not the true inverse. For perfect reconstruction,
        appropriate dual frame filters should be used.

    Example:
        >>> coeffs = torch.randn(1, 40, 250)
        >>> kernels = torch.randn(40, 128)
        >>> x_recon = circ_conv_transpose(coeffs, kernels, d=4)
    """
    L = y.shape[-1] * d
    y_up = upsample(y, d)

    kernels_long = F.pad(kernels, (0, L - kernels.shape[-1]), mode="constant", value=0)
    kernels_centered = torch.roll(kernels_long, shifts=-kernels.shape[-1] // 2, dims=-1)
    kernels_synth = torch.flip(torch.conj(kernels_centered), dims=(1,))

    y_fft = torch.fft.fft(y_up, n=L, dim=-1)
    k_fft = torch.fft.fft(kernels_synth, n=L, dim=-1)
    x_fft = y_fft * k_fft
    x = torch.fft.ifft(x_fft, dim=-1)
    x = torch.sum(x, dim=-2, keepdim=True)

    return torch.roll(x, 1, -1)
