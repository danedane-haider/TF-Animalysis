import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Tuple, Union


####################################################################################################
################### Routines for constructing auditory filterbanks #################################
####################################################################################################


def freqtoaud(
    freq: Union[float, int, torch.Tensor],
    scale: str = "erb",
    fs: Union[int, None] = None,
) -> torch.Tensor:
    """Convert frequencies from Hz to auditory scale units.

    Args:
        freq (Union[float, int, torch.Tensor]): Frequency value(s) in Hz
        scale (str): Auditory scale type. One of {'erb', 'mel', 'log10', 'elelog'}. Default: 'erb'
        fs (int, optional): Sampling frequency (required for 'elelog' scale). Default: None

    Returns:
        torch.Tensor: Corresponding auditory scale units

    Raises:
        ValueError: If unsupported scale is specified or fs is missing for 'elelog'

    Note:
        - ERB: Equivalent Rectangular Bandwidth (Glasberg & Moore)
        - MEL: Mel scale (perceptually uniform pitch)
        - Bark: Bark scale (critical band rate)
        - elelog: Logarithmic scale adapted for elephant hearing

    Example:
        >>> freq_hz = torch.tensor([100, 1000, 8000])
        >>> mel_units = freqtoaud(freq_hz, scale='mel')
    """

    scale = scale.lower()

    if isinstance(freq, (int, float)):
        freq = torch.tensor(freq)

    if scale == "erb":
        # Glasberg and Moore's ERB scale
        return 9.2645 * torch.sign(freq) * torch.log(1 + torch.abs(freq) * 0.00437)

    elif scale == "mel":
        # MEL scale
        return (
            1000
            / torch.log(torch.tensor(17 / 7))
            * torch.sign(freq)
            * torch.log(1 + torch.abs(freq) / 700)
        )

    elif scale == "log10":
        return torch.log10(torch.maximum(torch.ones(1), freq))

    elif scale == "elelog":
        if fs is None:
            raise ValueError(
                "Sampling frequency fs must be provided for 'elelog' scale."
            )
        fmin = 5
        fmax = 1000
        k = 0.88
        A = fmin / (1 - k)
        alpha = torch.log10(torch.tensor(fmax / A + k))
        return torch.log10(freq / A + k) / alpha

    else:
        raise ValueError(
            f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'log10', 'elelog'."
        )


def audtofreq(
    aud: Union[float, int, torch.Tensor],
    scale: str = "erb",
    fs: Union[int, None] = None,
) -> torch.Tensor:
    """Convert auditory scale units back to frequencies in Hz.

    Args:
        aud (Union[float, int, torch.Tensor]): Auditory scale values
        scale (str): Auditory scale type. One of {'erb', 'mel', 'log10', 'elelog'}. Default: 'erb'
        fs (int, optional): Sampling frequency (required for 'elelog' scale). Default: None

    Returns:
        torch.Tensor: Corresponding frequencies in Hz

    Example:
        >>> mel_units = torch.tensor([100, 1000, 2000])
        >>> freq_hz = audtofreq(mel_units, scale='mel')
    """
    if scale == "erb":
        return (1 / 0.00437) * (torch.exp(aud / 9.2645) - 1)

    elif scale == "mel":
        return (
            700
            * torch.sign(aud)
            * (torch.exp(torch.abs(aud) * torch.log(torch.tensor(17 / 7)) / 1000) - 1)
        )

    elif scale == "log10":
        return 10**aud

    elif scale == "elelog":
        if fs is None:
            raise ValueError(
                "Sampling frequency fs must be provided for 'elelog' scale."
            )
        fmin = 5
        fmax = 1000
        k = 0.88
        A = fmin / (1 - k)
        alpha = torch.log10(torch.tensor(fmax / A + k))
        return A * (10 ** (aud * alpha) - k)

    else:
        raise ValueError(
            f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'log10', 'elelog'."
        )


def audspace(
    fmin: Union[float, int, torch.Tensor],
    fmax: Union[float, int, torch.Tensor],
    num_channels: int,
    scale: str = "erb",
):
    """
    Computes a vector of num_channels many values equidistantly spaced on the selected auditory scale
    between fmin and fmax.

    Parameters:
        fmin (float): Minimum frequency in Hz.
        fmax (float): Maximum frequency in Hz.
        num_channels (int): Number of points in the output vector.
        audscale (str): Auditory scale (default is 'erb').
    Returns:
        tuple:
            y (ndarray): Array of frequencies equidistantly scaled on the auditory scale.
    """

    if num_channels <= 0:
        raise ValueError("n must be a positive integer scalar.")

    if fmin > fmax:
        raise ValueError("fmin must be less than or equal to fmax.")

    # Convert [fmin, fmax] to auditory scale
    if scale == "log10" or scale == "elelog":
        fmin = torch.maximum(torch.tensor(fmin), torch.ones(1))
    audlimits = freqtoaud(torch.tensor([fmin, fmax]), scale)

    # Generate frequencies spaced evenly on the auditory scale
    aud_space = torch.linspace(audlimits[0], audlimits[1], num_channels)
    y = audtofreq(aud_space, scale)

    # Ensure exact endpoints
    y[0] = fmin
    y[-1] = fmax

    return y


def freqtoaud_mod(
    freq: Union[float, int, torch.Tensor],
    fc_low: Union[float, int, torch.Tensor],
    fc_high: Union[float, int, torch.Tensor],
    scale="erb",
    fs=None,
):
    """
    Modified auditory scale function with linear regions below fc_low and above fc_high.

    Parameters:
        freq (ndarray): Frequency values in Hz.
        fc_low (float): Lower transition frequency in Hz.
        fc_high (float): Upper transition frequency in Hz.
    Returns:
        ndarray:
            Values on the modified auditory scale.
    """
    aud_crit_low = freqtoaud(fc_low, scale, fs)
    aud_crit_high = freqtoaud(fc_high, scale, fs)
    slope_low = (freqtoaud(fc_low * 1.01, scale, fs) - aud_crit_low) / (fc_low * 0.01)
    slope_high = (freqtoaud(fc_high * 1.01, scale, fs) - aud_crit_high) / (
        fc_high * 0.01
    )

    linear_low = freq < fc_low
    linear_high = freq > fc_high
    auditory = [not x for x in (linear_low + linear_high)]

    aud = torch.zeros_like(freq, dtype=torch.float32)

    aud[linear_low] = slope_low * (freq[linear_low] - fc_low) + aud_crit_low
    aud[auditory] = freqtoaud(freq[auditory], scale, fs)
    aud[linear_high] = slope_high * (freq[linear_high] - fc_high) + aud_crit_high

    return aud


def audtofreq_mod(
    aud: Union[float, int, torch.Tensor],
    fc_low: Union[float, int, torch.Tensor],
    fc_high: Union[float, int, torch.Tensor],
    scale="erb",
    fs=None,
):
    """
    Inverse of freqtoaud_mod to map auditory scale back to frequency.

    Parameters:
        aud (ndarray): Auditory scale values.
        fc_low (float): Lower transition frequency in Hz.
        fc_high (float): Upper transition frequency in Hz.
    Returns:
        ndarray:
            Frequency values in Hz
    """
    aud_crit_low = freqtoaud(fc_low, scale, fs)
    aud_crit_high = freqtoaud(fc_high, scale, fs)
    slope_low = (freqtoaud(fc_low * 1.01, scale, fs) - aud_crit_low) / (fc_low * 0.01)
    slope_high = (freqtoaud(fc_high * 1.01, scale, fs) - aud_crit_high) / (
        fc_high * 0.01
    )

    linear_low = aud < aud_crit_low
    linear_high = aud > aud_crit_high
    auditory_part = [not x for x in (linear_low + linear_high)]

    freq = torch.zeros_like(aud, dtype=torch.float32)

    freq[linear_low] = (aud[linear_low] - aud_crit_low) / slope_low + fc_low
    freq[auditory_part] = audtofreq(aud[auditory_part], scale, fs)
    freq[linear_high] = (aud[linear_high] - aud_crit_high) / slope_high + fc_high

    return freq


def audspace_mod(
    fc_low: Union[float, int, torch.Tensor],
    fc_high: Union[float, int, torch.Tensor],
    fs: int,
    num_channels: int,
    scale: str = "erb",
):
    """Generate num_channels many points that are equidistant in the modified auditory scale.

    Parameters:
        fc_crit (float): Critical frequency in Hz.
        fs (int): Sampling rate in Hz.
        M (int): Number of filters/channels.

    Returns:
        ndarray:
            Frequency values in Hz and in the auditory scale.
    """
    if fc_low > fc_high:
        raise ValueError("fc_low must be less than fc_high.")
    elif fc_low == fc_high:
        # equidistant samples form 0 to fs/2
        fc = torch.linspace(0, fs // 2, num_channels)
        return fc, freqtoaud_mod(fc, fs // 2, fs // 2, scale, fs)
    elif fc_low < fc_high:
        # Convert [0, fs//2] to modified auditory scale
        # aud_min = freqtoaud_mod(torch.tensor([0]), fc_low, fc_high, scale, fs)[0]
        aud_min = freqtoaud_mod(torch.tensor([fc_low]), fc_low, fc_high, scale, fs)[0]
        # aud_max = freqtoaud_mod(torch.tensor([fs // 2]), fc_low, fc_high, scale, fs)[0]
        aud_max = freqtoaud_mod(torch.tensor([fc_high]), fc_low, fc_high, scale, fs)[0]

        # Generate frequencies spaced evenly on the modified auditory scale
        fc_aud = torch.linspace(aud_min, aud_max, num_channels)

        # Convert back to frequency scale
        fc = audtofreq_mod(fc_aud, fc_low, fc_high, scale, fs)

        # Ensure exact endpoints
        # fc[0] = 0
        fc[0] = fc_low
        # fc[-1] = fs // 2
        fc[-1] = fc_high

        return fc, fc_aud
    else:
        raise ValueError("There is something wrong with fc_low and fc_high.")


def fctobw(fc: Union[float, int, torch.Tensor], scale="mel"):
    """
    Computes the critical bandwidth of a filter at a given center frequency.

    Parameters:
        fc (float or ndarray): Center frequency in Hz. Must be non-negative.
        audscale (str): Auditory scale. Supported values are:
                    - 'mel': Mel scale (default)
                    - 'erb': Equivalent Rectangular Bandwidth
                    - 'log10': Logarithmic scale

    Returns:
        ndarray or float:
            Critical bandwidth at each center frequency.
    """
    if isinstance(fc, (list, tuple, int, float)):
        fc = torch.tensor(fc)
    if not (isinstance(fc, (float, int, torch.Tensor)) and torch.all(fc >= 0)):
        raise ValueError("fc must be a non-negative scalar or array.")

    # Compute bandwidth based on the auditory scale
    if scale == "erb":
        bw = 24.7 + fc / 9.265
    elif scale == "mel":
        bw = torch.log10(torch.tensor(17 / 7)) * (700 + fc) / 1000
    elif scale == "log10":
        bw = fc
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return bw


def bwtofc(bw: Union[float, int, torch.Tensor], scale="mel"):
    """
    Computes the center frequency corresponding to a given critical bandwidth.

    Parameters:
        bw (float or ndarray): Critical bandwidth. Must be non-negative.
        scale (str): Auditory scale. Supported values are:
                 - 'mel': Mel scale
                 - 'erb': Equivalent Rectangular Bandwidth
                 - 'log10': Logarithmic scale

    Returns:
        ndarray or float:
            Center frequency corresponding to the given bandwidth.
    """
    if isinstance(bw, (list, tuple)):
        bw = torch.tensor(bw)
    if not (isinstance(bw, (float, int, torch.Tensor)) and torch.all(bw >= 0)):
        raise ValueError("bw must be a non-negative scalar or array.")

    # Compute center frequency based on the auditory scale
    if scale == "erb":
        fc = (bw - 24.7) * 9.265
    elif scale == "mel":
        fc = 1000 * (bw / torch.log10(torch.tensor(17 / 7))) - 700
    elif scale == "log10":
        fc = bw
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return fc


def firwin(kernel_size: int, padto: int = None):
    """
    FIR Hann window generation in Python.

    Parameters:
        kernel_size (int): Length of the window.
        padto (int): Length to which it should be padded.
        name (str): Name of the window.

    Returns:
        g (ndarray): FIR window.
    """
    g = torch.hann_window(kernel_size, periodic=False)
    g /= torch.sum(torch.abs(g))

    if padto is None or padto == kernel_size:
        return g
    elif padto > kernel_size:
        g_padded = torch.concatenate([g, torch.zeros(padto - len(g))])
        g_centered = torch.roll(g_padded, int((padto - len(g)) // 2))
        return g_centered
    else:
        raise ValueError("padto must be larger than kernel_size.")


def modulate(g: torch.Tensor, fc: Union[float, int, torch.Tensor], fs: int):
    """Modulate a filter by fc Hz.

    Args:
        g (list of torch.Tensor): Filters.
        fc (list): Center frequencies.
        fs (int): Sampling rate.

    Returns:
        g_mod (list of torch.Tensor): Modulated filters.
    """
    Lg = len(g)
    return g * torch.exp(2 * torch.pi * 1j * fc * torch.arange(Lg) / fs)


####################################################################################################
########################################### ISAC ###################################################
####################################################################################################


def audfilters(
    fs: int,
    kernel_size: Union[int, None] = None,
    num_channels: int = 96,
    fmin: Union[float, int] = 0,
    fmax: Union[float, int, None] = None,
    supp_mult: float = 1,
    scale: str = "elelog",
) -> Tuple[
    torch.Tensor,
    int,
    torch.Tensor,
    Union[int, float],
    Union[int, float],
    int,
    int,
    int,
    torch.Tensor,
]:
    """Generate auditory-inspired FIR filterbank kernels.

    Creates a bank of bandpass filters with center frequencies distributed according
    to perceptual auditory scales (mel, erb, elelog, etc.) and variable bandwidths.

    Args:
        fs (int): Sampling frequency in Hz. (required)
        kernel_size (int, optional): Length of the window for fmin. If None, computed automatically from fmin.
        num_channels (int): Number of frequency channels. Default: 96
        fmin (float, optional): Minimum center frequency in Hz. If None, uses 0. Default: None
        fmax (float, optional): Maximum center frequency in Hz. If None, uses fs//2. Default: None
        supp_mult (float): Support multiplier for kernel sizing above fmin. Values in [0,1].
        For supp_mult = 0 all windows have length equal to kernel_size.
        For supp_mult = 1  every window is 10 cycles of its associated center frequency long. Default: 1.0
        scale (str): Auditory scale. One of {'mel', 'erb', 'log10', 'elelog'}. Default: 'mel'

    Returns:
        Tuple containing:
            - kernels (torch.Tensor): Filter kernels of shape (num_channels, kernel_size)
            - d (int): Recommended stride for 50% overlap
            - fc (torch.Tensor): Center frequencies in Hz
            - Ls (int): Adjusted signal length
            - tsupp (torch.Tensor): Time support for each filter

    Raises:
        ValueError: If parameters are invalid (negative values, unsupported scale, etc.)

    Note:
        The filterbank construction follows auditory modeling principles where:
        - Low frequencies use longer filters (better frequency resolution)
        - High frequencies use shorter filters (better time resolution)

    Example:
        >>> kernels, stride, fc, Ls, _ = audfilters(
        ...     kernel_size=128, num_channels=40, fs=16000, scale='mel'
        ... )
        >>> print(f"Generated {kernels.shape[0]} filters with stride {stride}")
    """

    # check if all inputs are valid
    if kernel_size is not None and kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")
    if num_channels <= 0:
        raise ValueError("num_channels must be a positive integer.")
    # check if fs is a positive integer
    if fs is None:
        raise ValueError("sampling rate must be set.")
    if not isinstance(fs, int) or fs <= 0:
        raise ValueError("fs must be a positive integer.")
    if supp_mult < 0:
        raise ValueError("supp_mult must be a non-negative float.")
    if scale not in ["mel", "erb", "log10", "elelog"]:
        raise ValueError("scale must be one of 'mel', 'erb', 'log10', or 'elelog'.")
    if fmax is not None and (fmax <= 0 or fmax > fs // 2):
        raise ValueError("fmax must be a positive integer less than or equal to fs/2.")
    if fmax is None:
        fmax = fs // 2
    if fmin < 0 or fmin >= fmax:
        raise ValueError("fmin must be a non-negative integer less than fmax.")

    ####################################################################################################
    # Bandwidth conversion
    ####################################################################################################

    probeLs = 10000
    probeLg = 1000
    g_probe = firwin(probeLg, probeLs)

    # peak normalize
    gf_probe = torch.real(
        torch.fft.fft(g_probe) / torch.max(torch.abs(torch.fft.fft(g_probe)))
    )
    bw_probe = torch.norm(gf_probe) ** 2 * probeLg / probeLs / 2

    # preset bandwidth factors to get a good condition number
    if scale == "erb":
        bw_factor = 0.608
    elif scale == "mel":
        bw_factor = 111.33
    elif scale == "log10":
        bw_factor = 0.2
    elif scale == "elelog":
        bw_factor = 1.0

    bw_conversion = bw_probe / bw_factor  # * num_channels / 40

    ####################################################################################################
    # Center frequencies
    ####################################################################################################

    # checking the maximum kernel size
    if scale == "elelog":
        cycles = 10
        kernel_max = fs // 10 * cycles  # capture frequencies of 10Hz for 10 cycles

        if kernel_size is None:
            kernel_size = kernel_max

        if fmin is None:
            fmin = 5

        if fmax is None:
            fmax = 1000

        fc_min = fmin
        fc_max = fmax

        kernel_min = int(fs / fmax * cycles)
    else:
        fsupp_min = fctobw(0, scale)

        # if not specified, set the kernel size equal to the sampling frequency fs
        if kernel_size is None:
            kernel_size = int(
                torch.minimum(
                    torch.round(bw_conversion / fsupp_min * fs), torch.tensor(fs)
                )
            )

        # get the bandwidth for the kernel size and the associated center frequency
        fsupp_low = bw_conversion / kernel_size * fs
        fc_min = bwtofc(fsupp_low, scale)
        fc_max = fs // 2

        # get the bandwidth for the maximum center frequency and the associated kernel size
        fsupp_high = fctobw(fc_max, scale)
        kernel_min = int(torch.round(bw_conversion / fsupp_high * fs))

        if fc_min >= fc_max:
            fc_max = fc_min
            kernel_min = kernel_size
            Warning(
                f"fc_max was increased to {fc_min} to enable the kernel size of {kernel_size}."
            )

    # get center frequencies
    [fc, _] = audspace_mod(fc_min, fc_max, fs, num_channels, scale)

    num_low = torch.where(fc <= fc_min)[0].shape[0]
    num_high = torch.where(fc >= fc_max)[0].shape[0]
    num_aud = num_channels - num_low - num_high

    ####################################################################################################
    # Frequency and time supports
    ####################################################################################################

    # get time supports
    tsupp_low = (torch.ones(num_low) * kernel_size).int()
    tsupp_high = (torch.ones(num_high) * kernel_min).int()
    if scale == "elelog":
        # For each fc, interpolate between normal scaling (supp_mult=0) and max kernel size (supp_mult=1)
        fc_aud = fc[num_low : num_low + num_aud]

        # Normal scaling: fs / fc * cycles
        kernel_normal = fs / fc_aud * cycles

        # For fc <= fmax: interpolate towards kernel_size
        # For fc > fmax: stay at normal scaling
        kernel_target = torch.where(
            fc_aud <= fmax,
            torch.tensor(kernel_size, dtype=torch.float32),
            kernel_normal
        )

        # Linear interpolation based on supp_mult
        tsupp_aud_calc = (1 - supp_mult) * kernel_normal + supp_mult * kernel_target

        tsupp_aud = (
            torch.minimum(
                torch.tensor(kernel_size),
                torch.round(tsupp_aud_calc),
            )
        ).int()
        tsupp = torch.concatenate([tsupp_low, tsupp_aud, tsupp_high]).int()
    else:
        if num_low + num_high == num_channels:
            fsupp = fctobw(fc_max, scale)
            tsupp = tsupp_low
        else:
            fsupp = fctobw(fc[num_low : num_low + num_aud], scale)
            tsupp_aud = torch.round(bw_conversion / fsupp * fs)
            tsupp = torch.concatenate([tsupp_low, tsupp_aud, tsupp_high]).int()

    kernel_min = tsupp.min()
    kernel_size = tsupp.max()

    # Decimation factor (stride) for 50% overlap
    d = torch.maximum(kernel_min // 2, torch.tensor(1))
    #Ls = int(torch.ceil(L / d) * d)

    ####################################################################################################
    # Generate filters
    ####################################################################################################

    # only compute if the channel is lower than fmax
    num_channels_valid = int(sum(fc <= fmax))

    g = torch.zeros((num_channels_valid, kernel_size), dtype=torch.cfloat)

    g[0, :] = torch.sqrt(d) * firwin(kernel_size) / torch.sqrt(torch.tensor(2))

    for m in range(1, num_channels_valid):
        g[m, :] = torch.sqrt(d) * modulate(firwin(tsupp[m], kernel_size), fc[m], fs)

    Ls = 0 

    return g, int(d), fc, Ls, tsupp


####################################################################################################
####################################################################################################
####################################################################################################


def response(g: np.ndarray, fs: int) -> np.ndarray:
    """Compute frequency responses of filter kernels.

    Args:
        g (np.ndarray): Filter kernels of shape (num_channels, kernel_size)
        fs (int): Sampling frequency for frequency axis scaling

    Returns:
        np.ndarray: Magnitude-squared frequency responses of shape (2*num_channels, fs//2)

    Note:
        Computes responses for both analysis and conjugate filters.
    """
    g_full = np.concatenate([g, np.conj(g)], axis=0)
    G = np.abs(np.fft.fft(g_full, fs, axis=1)[:, : fs // 2]) ** 2

    return G


def plot_response(
    g: np.ndarray,
    fs: int,
    scale: str = "mel",
    plot_scale: bool = False,
    fc_min: Union[float, None] = None,
    fc_max: Union[float, None] = None,
    decoder: bool = False,
) -> None:
    """Plot frequency responses and auditory scale visualization of filters.

    Creates comprehensive visualization showing individual filter responses,
    total power spectral density, and optional auditory scale mapping.

    Args:
        g (np.ndarray): Filter kernels of shape (num_channels, kernel_size)
        fs (int): Sampling frequency in Hz for frequency axis scaling
        scale (str): Auditory scale name for scale plotting. Default: 'mel'
        plot_scale (bool): Whether to plot the auditory scale mapping. Default: False
        fc_min (float, optional): Lower transition frequency for scale visualization. Default: None
        fc_max (float, optional): Upper transition frequency for scale visualization. Default: None
        decoder (bool): Whether filters are for synthesis (affects plot titles). Default: False

    Note:
        This function displays plots and does not return values.
        Creates 2-3 subplots depending on plot_scale parameter.

    Example:
        >>> filters = np.random.randn(40, 128)
        >>> plot_response(filters, fs=16000, scale='mel', plot_scale=True)
    """
    num_channels = g.shape[0]

    g_hat = response(g, fs)
    g_hat_pos = g_hat[:num_channels, :]
    g_hat_pos[np.isnan(g_hat_pos)] = 0
    psd = np.sum(g_hat, axis=0)
    psd[np.isnan(psd)] = 0

    if plot_scale:
        plt.figure(figsize=(8, 2))
        freq_samples, _ = audspace_mod(fc_min, fc_max, fs, num_channels, scale)
        freqs = torch.linspace(0, fs // 2, fs // 2)

        auds = freqtoaud_mod(freqs, fc_min, fc_max, scale, fs).numpy()
        auds_orig = freqtoaud(freqs, scale, fs).numpy()

        plt.scatter(
            freq_samples.numpy(),
            freqtoaud_mod(freq_samples, fc_min, fc_max, scale, fs).numpy(),
            color="black",
            label="Center frequencies",
            linewidths=0.04,
        )
        plt.plot(freqs, auds, color="black", label=f"ISAC {scale}-scale")
        plt.plot(
            freqs,
            auds_orig,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"Original {scale}-scale",
        )

        if fc_min is not None:
            plt.axvline(fc_min, color="black", alpha=0.25)
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=0,
                x2=fc_min,
                color="gray",
                alpha=0.25,
            )
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=fc_min,
                x2=fs // 2,
                color="gray",
                alpha=0.1,
            )

        if fc_max is not None:
            plt.axvline(fc_max, color="black", alpha=0.25)
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=0,
                x2=fc_max,
                color="gray",
                alpha=0.25,
            )
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=fc_max,
                x2=fs // 2,
                color="gray",
                alpha=0.1,
            )

        plt.xlim([0, fs // 2])
        plt.ylim([auds[0] - 1, auds[-1] * 1.1])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Auditory Units")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

    fr_id = 0
    psd_id = 1

    f_range = np.linspace(0, fs // 2, fs // 2)
    ax[fr_id].set_xlim([0, fs // 2])
    ax[fr_id].set_ylim([0, np.max(g_hat_pos) * 1.1])
    ax[fr_id].plot(f_range, g_hat_pos.T)
    if decoder:
        ax[fr_id].set_title("PSDs of the synthesis filters")
    if not decoder:
        ax[fr_id].set_title("PSDs of the analysis filters")
    ax[fr_id].set_ylabel("Magnitude")

    ax[psd_id].plot(f_range, psd)
    ax[psd_id].set_xlim([0, fs // 2])
    ax[psd_id].set_ylim([0, np.max(psd) * 1.1])
    ax[psd_id].set_title("Total PSD")
    ax[psd_id].set_xlabel("Frequency [Hz]")
    ax[psd_id].set_ylabel("Magnitude")

    if fc_min is not None:
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=0, x2=fc_min, color="gray", alpha=0.25
        )
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=fc_min, x2=fs // 2, color="gray", alpha=0.1
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=0, x2=fc_min, color="gray", alpha=0.25
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=fc_min, x2=fs // 2, color="gray", alpha=0.1
        )

    if fc_max is not None:
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=0, x2=fc_max, color="gray", alpha=0.25
        )
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=fc_max, x2=fs // 2, color="gray", alpha=0.1
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=0, x2=fc_max, color="gray", alpha=0.25
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=fc_max, x2=fs // 2, color="gray", alpha=0.1
        )

    plt.tight_layout()
    plt.show()


# def ISACgram(
#     c: torch.Tensor,
#     fc: Union[torch.Tensor, None] = None,
#     L: Union[int, None] = None,
#     fs: Union[int, None] = None,
#     fmax: Union[float, None] = None,
#     log_scale: bool = False,
#     vmin: Union[float, None] = None,
#     cmap: str = "inferno",
# ) -> None:
#     """Plot time-frequency representation of filterbank coefficients.

#     Creates a spectrogram-like visualization with frequency on y-axis and time on x-axis.
#     Supports logarithmic scaling and frequency range limitation for better visualization.

#     Args:
#         c (torch.Tensor): Filterbank coefficients of shape (batch_size, num_channels, num_frames)
#         fc (torch.Tensor, optional): Center frequencies in Hz for y-axis labeling. Default: None
#         L (int, optional): Original signal length for time axis scaling. Default: None
#         fs (int, optional): Sampling frequency for time axis scaling. Default: None
#         fmax (float, optional): Maximum frequency to display in Hz. Default: None
#         log_scale (bool): Whether to apply log10 scaling to coefficients. Default: False
#         vmin (float, optional): Minimum value for dynamic range clipping. Default: None
#         cmap (str): Matplotlib colormap name. Default: 'inferno'

#     Note:
#         This function displays a plot and does not return values.
#         Only processes the first batch element if batch_size > 1.

#     Example:
#         >>> coeffs = torch.randn(1, 40, 250)
#         >>> fc = torch.linspace(100, 8000, 40)
#         >>> ISACgram(coeffs, fc=fc, L=16000, fs=16000, log_scale=True)
#     """
#     plt.figure(figsize=(10, 6))
#     ax = plt.gca()

#     c = c[0].detach().cpu().numpy()

#     if log_scale:
#         c = np.log10(np.abs(c) + 1e-10)

#     if fc is not None and fmax is not None:
#         c = c[: np.argmax(fc > fmax), :]

#     if vmin is not None:
#         mesh = ax.pcolor(c, cmap=cmap, vmin=np.min(c) * vmin)
#     else:
#         mesh = ax.pcolor(c, cmap=cmap)

#     # Add colorbar
#     plt.colorbar(mesh, ax=ax)

#     # Axis labeling
#     if fc is not None:
#         locs = np.linspace(0, c.shape[0] - 1, min(len(fc), 10)).astype(int)
#         ax.set_yticks(locs)
#         ax.set_yticklabels([int(np.round(fc[i])) for i in locs])

#         # X-axis: time
#         num_time_labels = 10
#         xticks = np.linspace(0, c.shape[1] - 1, num_time_labels)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(
#             [np.round(x, 1) for x in np.linspace(0, L // fs, num_time_labels)]
#         )

#         ax.set_ylabel("Frequency [Hz]")
#         ax.set_xlabel("Time [s]")
#     else:
#         ax.set_ylabel("Frequency index")
#         ax.set_xlabel("Time samples")

#     plt.tight_layout()
#     # plt.savefig('/Users/dani/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/ELECOM/IBAC/rumble_avg.png', dpi=600)
#     plt.show()