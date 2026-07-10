import torch

from .utils_auditory_scales import audtofreq, freqtoaud

######################################################################################################
#####################  ELE SPECTROGRAM ROUTINES ##########################################################################################
########################################################################################################


def create_triangular_filterbank(
    all_freqs: torch.Tensor,
    f_pts: torch.Tensor,
    drop_empty_filters: bool = True,
) -> torch.Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter` + 2) - includes edges.

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """

    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    # Optionally remove columns that are all zeros (can happen if fmax is too low).
    if drop_empty_filters:
        fb = fb[:, torch.sum(fb, dim=0) > 0]

    return fb



def melscale_fbanks(
    n_freqs: int,
    fmin: float,
    fmax: float,
    n_mels: int,
    sample_rate: int,
    norm: str = None,
    scale: str = 'elelog',
    drop_empty_filters: bool = True,
) -> torch.Tensor:
    """Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        fmin (float): Minimum frequency (Hz)
        fmax (float): Maximum frequency (Hz)
        num_channels (int): Number of mel filterbanks
        fs (int): Sample rate of the audio waveform
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        scale (str): Auditory scale to use ('elelog', 'mel', 'erb', 'log10'). Default: 'elelog'

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A @ melscale_fbanks(A.size(-1), ...)``.

    """

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')
    if fmax is None:
        fmax = sample_rate // 2

    # For elelog scale, minimum frequency must be >= 1 Hz
    if scale == 'elelog' and fmin < 1:
        fmin = 1

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate freq bins in the auditory scale
    m_min = freqtoaud(fmin, scale=scale, fs=sample_rate)
    m_max = freqtoaud(fmax, scale=scale, fs=sample_rate)
    fc = torch.linspace(m_min, m_max, n_mels + 2)

    f_pts = audtofreq(fc, scale=scale, fs=sample_rate)

    # For elelog scale, extend the first filter edge down to 0 Hz to ensure
    # it captures low-frequency FFT bins
    if scale == 'elelog' and fmin <= 1:
        f_pts[0] = 0.0
    # For other scales, can use perceptual bandwidth
    fb = create_triangular_filterbank(
        all_freqs,
        f_pts,
        drop_empty_filters=drop_empty_filters,
    )

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb