"""Automatic elephant-rumble F0 tracking in an Elelet representation.

The project annotations follow the first clearly visible spectral ridge (H1),
which is normally the second harmonic of the synthesis F0.  This module tracks
H1 and returns both H1 and ``F0 = H1 / 2`` explicitly.

The tracker has three deliberately small pieces:

1. local spectral whitening suppresses broadband wind/recording noise;
2. an Elelet-space SHR/SHRP score combines the H1 ridge with higher harmonics;
3. a restricted-band Viterbi path rejects isolated peaks without flattening FM.

No learned model or reference contour is used during inference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


@dataclass(frozen=True)
class HybridEleletConfig:
    """Configuration for :func:`track_hybrid_elelet`."""

    sr: int = 16_000
    hop_length: int = 256
    kernel_size: int = 16_000
    supp_mult: float = 0.3
    num_channels: int = 1024
    transform_fmin: float = 5.0
    transform_fmax: float = 500.0
    scale: str = "elelog"
    channel_batch_size: int | None = 64

    # H1 is the annotated ridge.  Synthesis F0 is half these values.
    h1_min: float = 17.0
    h1_max: float = 82.0
    candidate_step_hz: float = 0.1
    selected_band_width_hz: float = 36.0  # retained for the profile-selector utility
    proposal_bands_hz: tuple[tuple[float, float], ...] = (
        (17.0, 30.0),
        (22.5, 40.0),
        (30.0, 58.0),
        (50.0, 82.0),
    )
    proposal_anchor_threshold_hz: float = 22.5
    proposal_max_jump_hz: float = 1.5
    proposal_ridge_weight: float = 0.35
    refinement_radius_hz: float = 0.5

    n_harmonics: int = 24
    harmonic_weight_exp: float = 0.35
    harmonic_bin_radius: int = 1
    ridge_weight: float = 0.65
    interharmonic_penalty: float = 0.30
    contrast_sigma_bins: float = 8.0

    # Emissions are robustly standardized per frame.  This transition cost is
    # therefore intentionally modest and preserves sub-Hz modulation.
    smoothness: float = 0.04
    transition_scale_hz: float = 0.1
    max_jump_hz: float = 1.5


@dataclass(frozen=True)
class HybridEleletResult:
    time: np.ndarray
    h1_hz: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    proposal_confidence: float
    selected_band_hz: tuple[float, float]
    candidates_h1_hz: np.ndarray
    emission: np.ndarray
    ridge_emission: np.ndarray
    shrp_emission: np.ndarray


def make_elelet(config: HybridEleletConfig):
    """Construct the Elelet transform used by the hybrid tracker."""
    from tf_transforms.transforms import Elelet

    return Elelet(
        kernel_size=config.kernel_size,
        num_channels=config.num_channels,
        stride=config.hop_length,
        f_min=config.transform_fmin,
        f_max=config.transform_fmax,
        fs=config.sr,
        supp_mult=config.supp_mult,
        scale=config.scale,
        use_torch=False,
        backend="fft_decimated",
        channel_batch_size=config.channel_batch_size,
        cache_kernel_fft=False,
    )


def compute_elelet_magnitude(
    audio: np.ndarray,
    config: HybridEleletConfig,
    *,
    transform=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return magnitude, center frequencies, and frame times."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono or samples-by-channels; got {audio.shape}")

    transform = transform if transform is not None else make_elelet(config)
    magnitude = np.abs(transform(audio)).astype(np.float64, copy=False)
    frequencies = np.asarray(transform.fc, dtype=np.float64)[: magnitude.shape[0]]
    times = np.arange(magnitude.shape[1], dtype=np.float64) * config.hop_length / config.sr
    return magnitude, frequencies, times


def local_spectral_contrast(magnitude: np.ndarray, sigma_bins: float = 8.0) -> np.ndarray:
    """Whiten each frame against its smooth local spectral envelope.

    Subtracting a frequency-smoothed log spectrum is a cheap form of spectral
    whitening.  Broadband wind and microphone colour largely disappear while
    narrow harmonic ridges remain positive.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if magnitude.ndim != 2:
        raise ValueError("magnitude must have shape (frequency, time)")

    positive = magnitude[magnitude > 0.0]
    eps = (float(np.median(positive)) * 1e-6) if positive.size else 1e-12
    log_magnitude = np.log(np.maximum(magnitude, eps))
    envelope = gaussian_filter1d(log_magnitude, sigma=max(float(sigma_bins), 0.5), axis=0, mode="nearest")
    contrast = log_magnitude - envelope

    # Keep frames numerically comparable, but do not let a silent frame create
    # arbitrarily large standardized noise values.
    scale = np.percentile(np.abs(contrast), 75.0, axis=0)
    floor = max(float(np.median(scale)) * 0.1, 1e-6)
    contrast /= np.maximum(scale, floor)[None, :]
    return np.clip(contrast, -4.0, 6.0)


def _sample_frequency_rows(
    values: np.ndarray,
    frequencies_hz: np.ndarray,
    targets_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly sample all time frames at fixed target frequencies."""
    targets_hz = np.asarray(targets_hz, dtype=np.float64)
    valid = (targets_hz >= frequencies_hz[0]) & (targets_hz <= frequencies_hz[-1])
    clipped = np.clip(targets_hz, frequencies_hz[0], frequencies_hz[-1])
    right = np.searchsorted(frequencies_hz, clipped, side="left")
    right = np.clip(right, 1, len(frequencies_hz) - 1)
    left = right - 1
    denom = frequencies_hz[right] - frequencies_hz[left]
    alpha = np.divide(
        clipped - frequencies_hz[left],
        denom,
        out=np.zeros_like(clipped),
        where=denom > 0.0,
    )
    sampled = (1.0 - alpha)[:, None] * values[left, :] + alpha[:, None] * values[right, :]
    sampled[~valid, :] = 0.0
    return sampled, valid


def elelet_shrp_components(
    magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    config: HybridEleletConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build H1-ridge and Elelet-SHRP emissions on a uniform H1 grid."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
    order = np.argsort(frequencies_hz)
    frequencies_hz = frequencies_hz[order]
    sorted_magnitude = magnitude[order, :]
    contrast = local_spectral_contrast(sorted_magnitude, config.contrast_sigma_bins)

    candidates = np.arange(
        config.h1_min,
        config.h1_max + 0.5 * config.candidate_step_hz,
        config.candidate_step_hz,
        dtype=np.float64,
    )
    ridge, _ = _sample_frequency_rows(contrast, frequencies_hz, candidates)
    positive = sorted_magnitude[sorted_magnitude > 0.0]
    eps = (float(np.median(positive)) * 1e-6) if positive.size else 1e-12
    log_magnitude = np.log(np.maximum(sorted_magnitude, eps))
    anchor, _ = _sample_frequency_rows(log_magnitude, frequencies_hz, candidates)

    f0_candidates = 0.5 * candidates
    harmonic_sum = np.zeros_like(ridge)
    interharmonic_sum = np.zeros_like(ridge)
    harmonic_weight = np.zeros(len(candidates), dtype=np.float64)
    interharmonic_weight = np.zeros(len(candidates), dtype=np.float64)

    # k=2 is H1.  k=1 is deliberately omitted because the sub-20-Hz region is
    # often dominated by wind/ground noise and the fundamental can be weak.
    for harmonic in range(2, config.n_harmonics + 1):
        weight = 1.0 / (float(harmonic) ** config.harmonic_weight_exp)
        targets = harmonic * f0_candidates
        sampled, valid = _sample_frequency_rows(contrast, frequencies_hz, targets)
        harmonic_sum += weight * sampled
        harmonic_weight += weight * valid

        # SHR-style valleys midway between expected partials penalize broadband
        # noise and octave/subharmonic alternatives.
        inter_targets = (harmonic + 0.5) * f0_candidates
        inter_sampled, inter_valid = _sample_frequency_rows(contrast, frequencies_hz, inter_targets)
        interharmonic_sum += weight * inter_sampled
        interharmonic_weight += weight * inter_valid

    harmonic_mean = harmonic_sum / np.maximum(harmonic_weight[:, None], 1e-12)
    interharmonic_mean = interharmonic_sum / np.maximum(interharmonic_weight[:, None], 1e-12)
    shrp = harmonic_mean - config.interharmonic_penalty * interharmonic_mean
    # Put the two cues on comparable scales before combining them.  The ridge
    # says which visible partial is H1; SHRP says whether its implied F0 has a
    # complete harmonic stack.
    ridge_z = robust_standardize_emission(ridge)
    shrp_z = robust_standardize_emission(shrp)
    combined = config.ridge_weight * ridge_z + shrp_z
    return candidates, combined, ridge, shrp, anchor


def robust_standardize_emission(emission: np.ndarray) -> np.ndarray:
    """Robustly standardize each frame across candidate frequencies."""
    center = np.median(emission, axis=0)
    mad = np.median(np.abs(emission - center[None, :]), axis=0)
    floor = max(float(np.median(mad)) * 0.1, 1e-5)
    standardized = (emission - center[None, :]) / np.maximum(1.4826 * mad, floor)[None, :]
    return np.clip(standardized, -8.0, 12.0)


def elelet_harmonic_power_emission(
    magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    candidates_h1_hz: np.ndarray,
    config: HybridEleletConfig,
) -> np.ndarray:
    """Existing harmonic-energy refinement adapted to an automatic H1 path.

    Unlike the contrast-based SHRP cue used for the coarse octave choice, this
    score uses raw Elelet power.  Once the search is restricted to a narrow
    radius, the displaced peaks of higher harmonics provide sub-Hz FM detail
    without a realistic risk of selecting another octave.
    """
    order = np.argsort(frequencies_hz)
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)[order]
    power = np.square(np.asarray(magnitude, dtype=np.float64)[order])
    radius = max(0, int(config.harmonic_bin_radius))
    if radius:
        power = uniform_filter1d(power, size=2 * radius + 1, axis=0, mode="nearest")

    f0_candidates = 0.5 * np.asarray(candidates_h1_hz, dtype=np.float64)
    score = np.zeros((len(candidates_h1_hz), power.shape[1]), dtype=np.float64)
    weight_sum = np.zeros(len(candidates_h1_hz), dtype=np.float64)
    for harmonic in range(2, config.n_harmonics + 1):
        weight = 1.0 / (float(harmonic) ** config.harmonic_weight_exp)
        sampled, valid = _sample_frequency_rows(
            power,
            frequencies_hz,
            harmonic * f0_candidates,
        )
        score += weight * sampled
        weight_sum += weight * valid
    score /= np.maximum(weight_sum[:, None], 1e-12)
    positive = score[score > 0.0]
    eps = (float(np.median(positive)) * 1e-8) if positive.size else 1e-12
    return np.log(np.maximum(score, eps))


def select_restricted_band(
    candidates_hz: np.ndarray,
    emission: np.ndarray,
    width_hz: float,
    *,
    anchor_emission: np.ndarray | None = None,
    shrp_emission: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Select an H1 interval around a ridge/SHRP-consistent anchor.

    A strong low-frequency ridge can be either H1 or the physical fundamental.
    We therefore compare the dominant ridge, half of it, and twice it using the
    SHRP profile.  This tiny octave test is more reliable than integrating a
    wide salience map, which tends to select an arbitrary upper partial.
    """
    if width_hz <= 0.0 or width_hz >= float(candidates_hz[-1] - candidates_hz[0]):
        return np.ones(len(candidates_hz), dtype=bool), (float(candidates_hz[0]), float(candidates_hz[-1]))

    profile = np.quantile(emission, 0.75, axis=1)
    if anchor_emission is None or shrp_emission is None:
        center_idx = int(np.argmax(profile))
    else:
        anchor_profile = np.quantile(anchor_emission, 0.75, axis=1)
        anchor_idx = int(np.argmax(anchor_profile))
        anchor_hz = float(candidates_hz[anchor_idx])
        hypotheses_hz = [0.5 * anchor_hz, anchor_hz, 2.0 * anchor_hz]
        hypothesis_idx = sorted(
            {
                int(np.argmin(np.abs(candidates_hz - value)))
                for value in hypotheses_hz
                if candidates_hz[0] <= value <= candidates_hz[-1]
            }
        )
        shrp_profile = np.quantile(robust_standardize_emission(shrp_emission), 0.75, axis=1)
        anchor_z = (anchor_profile - np.median(anchor_profile)) / max(
            1.4826 * float(np.median(np.abs(anchor_profile - np.median(anchor_profile)))),
            1e-6,
        )
        hypothesis_scores = shrp_profile[hypothesis_idx] + 0.35 * anchor_z[hypothesis_idx]
        center_idx = int(hypothesis_idx[int(np.argmax(hypothesis_scores))])
    half = 0.5 * width_hz
    center = float(candidates_hz[center_idx])
    low = max(float(candidates_hz[0]), center - half)
    high = min(float(candidates_hz[-1]), center + half)
    mask = (candidates_hz >= low) & (candidates_hz <= high)
    return mask, (float(candidates_hz[mask][0]), float(candidates_hz[mask][-1]))


def _coarse_peak_path(
    magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    band_hz: tuple[float, float],
    max_jump_hz: float,
) -> np.ndarray:
    """Track one visible ridge outward from the call's strongest frame."""
    mask = (frequencies_hz >= band_hz[0]) & (frequencies_hz <= band_hz[1])
    if np.count_nonzero(mask) < 3:
        raise ValueError(f"too few Elelet channels in proposal band {band_hz}")
    values = magnitude[mask]
    frequencies = frequencies_hz[mask]
    n_frames = values.shape[1]
    path = np.empty(n_frames, dtype=np.float64)

    global_idx = int(np.argmax(np.mean(values, axis=1)))
    global_hz = float(frequencies[global_idx])
    start = int(np.argmax(np.sum(values, axis=0)))

    def choose(frame: int, previous_hz: float) -> float:
        allowed = np.abs(frequencies - previous_hz) <= max_jump_hz
        if not np.any(allowed):
            return previous_hz
        score = np.where(allowed, values[:, frame], -np.inf)
        return float(frequencies[int(np.argmax(score))])

    path[start] = choose(start, global_hz)
    for frame in range(start + 1, n_frames):
        path[frame] = choose(frame, path[frame - 1])
    for frame in range(start - 1, -1, -1):
        path[frame] = choose(frame, path[frame + 1])
    return path


def _sample_path(emission: np.ndarray, candidates_hz: np.ndarray, path_hz: np.ndarray) -> np.ndarray:
    """Sample a candidate-by-time emission along a time-varying path."""
    right = np.searchsorted(candidates_hz, path_hz, side="left")
    right = np.clip(right, 1, len(candidates_hz) - 1)
    left = right - 1
    denom = candidates_hz[right] - candidates_hz[left]
    alpha = np.divide(
        path_hz - candidates_hz[left],
        denom,
        out=np.zeros_like(path_hz),
        where=denom > 0.0,
    )
    frames = np.arange(len(path_hz))
    return (1.0 - alpha) * emission[left, frames] + alpha * emission[right, frames]


def select_coarse_peak_proposal(
    magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    candidates_hz: np.ndarray,
    anchor_emission: np.ndarray,
    shrp_emission: np.ndarray,
    config: HybridEleletConfig,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Generate three restricted-band ridge paths and rank them with SHRP.

    This is the coarse octave decision.  SHRP never gets to invent an arbitrary
    contour: it only chooses among ridges that are visibly present in a low,
    normal, or high H1 band.
    """
    anchor_z = robust_standardize_emission(anchor_emission)
    shrp_z = robust_standardize_emission(shrp_emission)
    proposals: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []

    for low, high in config.proposal_bands_hz:
        low = max(float(low), config.h1_min)
        high = min(float(high), config.h1_max)
        if high <= low:
            continue
        path = _coarse_peak_path(
            magnitude,
            frequencies_hz,
            (low, high),
            max_jump_hz=config.proposal_max_jump_hz,
        )
        # Avoid giving duplicated paths extra votes in overlapping bands.
        if any(float(np.median(np.abs(path - previous))) < 0.5 for previous in proposals):
            continue
        shrp_score = float(np.median(_sample_path(shrp_z, candidates_hz, path)))
        ridge_score = float(np.median(_sample_path(anchor_z, candidates_hz, path)))
        roughness = float(np.median(np.abs(np.diff(path)))) if len(path) > 1 else 0.0
        score = shrp_score + config.proposal_ridge_weight * ridge_score - 0.05 * roughness
        proposals.append(path)
        diagnostics.append(
            {
                "band_min_hz": low,
                "band_max_hz": high,
                "median_h1_hz": float(np.median(path)),
                "shrp_score": shrp_score,
                "ridge_score": ridge_score,
                "roughness_hz": roughness,
                "score": score,
            }
        )

    if not proposals:
        raise ValueError("no valid coarse H1 proposals")
    broad_mask = (frequencies_hz >= config.h1_min) & (frequencies_hz <= config.h1_max)
    broad_frequencies = frequencies_hz[broad_mask]
    broad_magnitude = magnitude[broad_mask]
    broad_anchor_hz = float(broad_frequencies[int(np.argmax(np.mean(broad_magnitude, axis=1)))])
    if broad_anchor_hz >= config.proposal_anchor_threshold_hz:
        # In the usual case the dominant ridge is already inside the established
        # H1 band.  Preserve that identity; otherwise a strong upper harmonic can
        # win merely because its SHRP score is cleaner.
        best = int(
            np.argmin(
                [abs(item["median_h1_hz"] - broad_anchor_hz) for item in diagnostics]
            )
        )
    else:
        # Below 22.5 Hz the ridge may be either a genuinely low H1 or the physical
        # F0 of a higher call.  This is the only ambiguous case where SHRP decides.
        best = int(np.argmax([item["score"] for item in diagnostics]))
    return proposals[best], diagnostics


def viterbi_smooth_path(
    candidates_hz: np.ndarray,
    emission: np.ndarray,
    *,
    smoothness: float,
    transition_scale_hz: float,
    max_jump_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a maximum-score path with a bounded quadratic transition cost."""
    candidates_hz = np.asarray(candidates_hz, dtype=np.float64)
    emission = np.asarray(emission, dtype=np.float64)
    n_states, n_frames = emission.shape
    if n_states < 3 or n_frames == 0:
        raise ValueError("Viterbi tracking requires at least 3 states and 1 frame")

    step = float(np.median(np.diff(candidates_hz)))
    max_offset = max(1, int(np.floor(max_jump_hz / step + 1e-9)))
    offsets = np.arange(-max_offset, max_offset + 1, dtype=np.int32)
    delta_hz = offsets.astype(np.float64) * step
    costs = smoothness * np.square(delta_hz / max(transition_scale_hz, 1e-9))

    dp = emission[:, 0].copy()
    backpointers = np.zeros((n_frames, n_states), dtype=np.int16 if n_states < 32767 else np.int32)
    state_idx = np.arange(n_states)

    for frame in range(1, n_frames):
        alternatives = np.full((len(offsets), n_states), -np.inf, dtype=np.float64)
        predecessors = np.empty((len(offsets), n_states), dtype=np.int32)
        for oi, (offset, cost) in enumerate(zip(offsets, costs)):
            source = state_idx - int(offset)
            valid = (source >= 0) & (source < n_states)
            predecessors[oi] = np.clip(source, 0, n_states - 1)
            alternatives[oi, valid] = dp[source[valid]] - cost
        best_offset = np.argmax(alternatives, axis=0)
        best_predecessor = predecessors[best_offset, state_idx]
        dp = emission[:, frame] + alternatives[best_offset, state_idx]
        backpointers[frame] = best_predecessor

    path_idx = np.empty(n_frames, dtype=np.int32)
    path_idx[-1] = int(np.argmax(dp))
    for frame in range(n_frames - 1, 0, -1):
        path_idx[frame - 1] = int(backpointers[frame, path_idx[frame]])

    path_hz = candidates_hz[path_idx].copy()
    # Quadratic interpolation extracts sub-grid FM detail from the combined
    # higher-harmonic score without weakening the temporal path constraint.
    for frame, idx in enumerate(path_idx):
        if idx <= 0 or idx >= n_states - 1:
            continue
        left, center, right = emission[idx - 1 : idx + 2, frame]
        denom = left - 2.0 * center + right
        if np.isfinite(denom) and abs(denom) > 1e-12:
            offset = np.clip(0.5 * (left - right) / denom, -0.75, 0.75)
            path_hz[frame] += float(offset) * step
    return path_hz, path_idx


def path_confidence(
    candidates_hz: np.ndarray,
    emission: np.ndarray,
    path_hz: np.ndarray,
    exclusion_hz: float = 0.75,
) -> np.ndarray:
    """Return a 0--1 local margin confidence for a tracked path."""
    confidence = np.zeros(emission.shape[1], dtype=np.float64)
    for frame, frequency in enumerate(path_hz):
        selected_idx = int(np.argmin(np.abs(candidates_hz - frequency)))
        selected = float(emission[selected_idx, frame])
        alternatives = np.abs(candidates_hz - frequency) >= exclusion_hz
        rival = float(np.max(emission[alternatives, frame])) if np.any(alternatives) else selected
        margin = np.clip(selected - rival, -12.0, 12.0)
        confidence[frame] = 1.0 / (1.0 + np.exp(-margin))
    return confidence


def track_from_magnitude(
    magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    times: np.ndarray,
    config: HybridEleletConfig,
    *,
    component: str = "power",
) -> HybridEleletResult:
    """Track a contour from an already computed Elelet magnitude."""
    candidates, combined, ridge, shrp, anchor = elelet_shrp_components(magnitude, frequencies_hz, config)
    harmonic_power = elelet_harmonic_power_emission(magnitude, frequencies_hz, candidates, config)
    if component == "hybrid":
        chosen = combined
    elif component == "ridge":
        chosen = ridge
    elif component == "shrp":
        chosen = shrp
    elif component == "power":
        chosen = harmonic_power
    elif component == "hybrid_power":
        chosen = (
            config.ridge_weight * robust_standardize_emission(ridge)
            + robust_standardize_emission(harmonic_power)
        )
    else:
        raise ValueError("component must be one of: hybrid, ridge, shrp, power, hybrid_power")

    standardized = robust_standardize_emission(chosen)
    coarse_path, proposal_diagnostics = select_coarse_peak_proposal(
        magnitude,
        np.asarray(frequencies_hz, dtype=np.float64),
        candidates,
        anchor,
        shrp,
        config,
    )
    allowed = np.abs(candidates[:, None] - coarse_path[None, :]) <= config.refinement_radius_hz
    constrained_emission = np.where(allowed, standardized, -1e6)
    path_h1, _ = viterbi_smooth_path(
        candidates,
        constrained_emission,
        smoothness=config.smoothness,
        transition_scale_hz=config.transition_scale_hz,
        max_jump_hz=config.max_jump_hz,
    )
    local_confidence = path_confidence(candidates, standardized, path_h1)
    selected_proposal = int(
        np.argmin(
            [
                abs(item["median_h1_hz"] - float(np.median(coarse_path)))
                for item in proposal_diagnostics
            ]
        )
    )
    selected_score = float(proposal_diagnostics[selected_proposal]["score"])
    rival_scores = [
        float(item["score"])
        for index, item in enumerate(proposal_diagnostics)
        if index != selected_proposal
    ]
    proposal_margin = selected_score - max(rival_scores) if rival_scores else 12.0
    proposal_confidence = float(1.0 / (1.0 + np.exp(-np.clip(proposal_margin, -12.0, 12.0))))
    confidence = local_confidence * proposal_confidence
    band_low = max(config.h1_min, float(np.min(coarse_path)) - config.refinement_radius_hz)
    band_high = min(config.h1_max, float(np.max(coarse_path)) + config.refinement_radius_hz)
    band_mask = (candidates >= band_low) & (candidates <= band_high)
    band = (float(candidates[band_mask][0]), float(candidates[band_mask][-1]))
    return HybridEleletResult(
        time=np.asarray(times, dtype=np.float64)[: len(path_h1)],
        h1_hz=path_h1,
        f0_hz=0.5 * path_h1,
        confidence=confidence,
        proposal_confidence=proposal_confidence,
        selected_band_hz=band,
        candidates_h1_hz=candidates[band_mask],
        emission=constrained_emission[band_mask],
        ridge_emission=robust_standardize_emission(ridge)[band_mask],
        shrp_emission=robust_standardize_emission(shrp)[band_mask],
    )


def track_hybrid_elelet(
    audio: np.ndarray,
    config: HybridEleletConfig = HybridEleletConfig(),
    *,
    transform=None,
    component: str = "power",
) -> HybridEleletResult:
    """Compute Elelet coefficients and track H1/F0 from audio."""
    magnitude, frequencies, times = compute_elelet_magnitude(audio, config, transform=transform)
    return track_from_magnitude(magnitude, frequencies, times, config, component=component)


def with_parameters(config: HybridEleletConfig, **parameters) -> HybridEleletConfig:
    """Small public helper used by the tuning script."""
    return replace(config, **parameters)
