import numpy as np
import torch
import torch.nn.functional as F

from .utils_auditory_scales import audspace_mod, freqtoaud_mod, freqtoaud

####################################################################################################
#####################         ELELET Routines                      #################################
####################################################################################################

def circ_conv_numpy(x: np.ndarray, kernels: np.ndarray, d: int = 1) -> np.ndarray:
    L = x.shape[-1]
    kernel_size = kernels.shape[-1]

    # If audio is shorter than kernels, pad the audio first
    if L < kernel_size:
        pad_size = kernel_size - L
        x = np.pad(x, (0, pad_size), mode='constant', constant_values=0)
        L = x.shape[-1]

    # Zero-pad kernels to signal length
    kernels_long = np.pad(kernels, ((0, 0), (0, L - kernels.shape[-1])), mode='constant', constant_values=0)

    # Center the kernels by rolling
    kernels_centered = np.roll(kernels_long, shift=-kernels.shape[-1] // 2, axis=-1)

    # FFT-based circular convolution
    x_fft = np.fft.fft(x, n=L)
    k_fft = np.fft.fft(kernels_centered, n=L, axis=-1)
    y_fft = x_fft * k_fft
    y = np.fft.ifft(y_fft, axis=-1)

    # Downsample
    return y[:, ::d]

def upsample(x: torch.Tensor, d: int) -> torch.Tensor:
    N = x.shape[-1] * d
    x_up = F.pad(torch.zeros_like(x), (0, N - x.shape[-1]))
    x_up[:, :, ::d] = x
    return x_up

def circ_conv(x: torch.Tensor, kernels: torch.Tensor, d: int = 1) -> torch.Tensor:
    """Circular convolution with optional downsampling.

    Performs efficient circular convolution using FFT, followed by downsampling.
    The kernels are automatically centered for proper phase alignment.

    Args:
        x (torch.Tensor): Input signal of shape (..., signal_length)
        kernels (torch.Tensor): Filter kernels of shape (num_channels, 1, kernel_length)
            or (num_channels, kernel_length)
        d (int): Downsampling factor (stride). Default: 1

    Returns:
        torch.Tensor: Convolved and downsampled output of shape (..., num_channels, output_length)

    Note:
        Uses circular convolution which assumes periodic boundary conditions.
        Kernels are automatically zero-padded and centered.

    Example:
        >>> x = torch.randn(1, 1000)
        >>> kernels = torch.randn(40, 128)
        >>> y = circ_conv(x, kernels, d=4)
    """
    L = x.shape[-1]
    x = x.to(kernels.dtype)
    kernels = kernels.to(x.device)

    kernels_long = F.pad(kernels, (0, L - kernels.shape[-1]), mode="constant", value=0)
    kernels_centered = torch.roll(kernels_long, shifts=-kernels.shape[-1] // 2, dims=-1)

    x_fft = torch.fft.fft(x, n=L, dim=-1)
    k_fft = torch.fft.fft(kernels_centered, n=L, dim=-1)
    y_fft = x_fft * k_fft
    y = torch.fft.ifft(y_fft)

    return y[:, :, ::d]


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
