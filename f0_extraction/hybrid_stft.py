"""Multi-resolution STFT version of the automatic elephant-rumble tracker.

The algorithm is intentionally isomorphic to :mod:`hybrid_elelet`:

* one STFT supplies stable restricted-band H1 ridge proposals;
* a second, optionally shorter STFT supplies SHRP and harmonic-power evidence;
* the chosen proposal is refined locally with the same smooth Viterbi path.

The two actual window lengths matter; zero padding only densifies the sampled
frequency grid and is not treated as additional physical resolution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import librosa
import numpy as np

from f0_extraction.hybrid_elelet import (
    _coarse_peak_path,
    _sample_path,
    elelet_harmonic_power_emission,
    elelet_shrp_components,
    path_confidence,
    robust_standardize_emission,
    viterbi_smooth_path,
)


@dataclass(frozen=True)
class HybridStftConfig:
    sr: int = 16_000
    hop_length: int = 256

    # The two stages can use different physical windows. The evaluated default
    # uses the robust equal-window corner; 12000/8192 is the long-proposal preset.
    coarse_n_fft: int = 32_768
    coarse_win_length: int = 8_192
    fine_n_fft: int = 32_768
    fine_win_length: int = 8_192
    analysis_fmin: float = 5.0
    analysis_fmax: float = 500.0

    h1_min: float = 17.0
    h1_max: float = 82.0
    candidate_step_hz: float = 0.1
    proposal_bands_hz: tuple[tuple[float, float], ...] = (
        (17.0, 30.0),
        (22.5, 40.0),
        (30.0, 58.0),
        (50.0, 82.0),
    )
    proposal_anchor_threshold_hz: float = 22.5
    proposal_max_jump_hz: float = 1.5
    proposal_ridge_weight: float = 0.35
    refinement_radius_hz: float = 1.0

    n_harmonics: int = 24
    harmonic_weight_exp: float = 0.35
    harmonic_bin_radius: int = 2
    ridge_weight: float = 0.65
    interharmonic_penalty: float = 0.30
    contrast_sigma_bins: float = 8.0

    smoothness: float = 0.04
    transition_scale_hz: float = 0.1
    max_jump_hz: float = 1.5


@dataclass(frozen=True)
class HybridStftResult:
    time: np.ndarray
    h1_hz: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    proposal_confidence: float
    selected_band_hz: tuple[float, float]
    coarse_path_h1_hz: np.ndarray
    candidates_h1_hz: np.ndarray
    emission: np.ndarray


def _validate_config(config: HybridStftConfig) -> None:
    for label, n_fft, win_length in (
        ("coarse", config.coarse_n_fft, config.coarse_win_length),
        ("fine", config.fine_n_fft, config.fine_win_length),
    ):
        if n_fft <= 0 or win_length <= 0:
            raise ValueError(f"{label} n_fft and win_length must be positive")
        if win_length > n_fft:
            raise ValueError(f"{label}_win_length cannot exceed {label}_n_fft")
    if config.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if config.analysis_fmax <= config.analysis_fmin:
        raise ValueError("analysis_fmax must be greater than analysis_fmin")


def _stft_magnitude(
    audio: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    n_fft: int,
    win_length: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Centered STFT padding is intentional for short calls; librosa warns even
    # though the requested behaviour is well-defined.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"n_fft=.* is too large for input signal.*",
            category=UserWarning,
        )
        spectrum = librosa.stft(
            y=audio,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window="hann",
            center=True,
            pad_mode="constant",
        )
    magnitude = np.abs(spectrum).astype(np.float64, copy=False)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid = (frequencies >= fmin) & (frequencies <= fmax)
    times = np.arange(magnitude.shape[1], dtype=np.float64) * hop_length / sr
    return magnitude[valid], frequencies[valid], times


def compute_multires_stft(
    audio: np.ndarray,
    config: HybridStftConfig = HybridStftConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coarse/fine magnitudes, axes, and their common frame times."""
    _validate_config(config)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono or samples-by-channels; got {audio.shape}")

    coarse, coarse_frequencies, coarse_times = _stft_magnitude(
        audio,
        sr=config.sr,
        hop_length=config.hop_length,
        n_fft=config.coarse_n_fft,
        win_length=config.coarse_win_length,
        fmin=config.analysis_fmin,
        fmax=config.analysis_fmax,
    )
    if (
        config.coarse_n_fft == config.fine_n_fft
        and config.coarse_win_length == config.fine_win_length
    ):
        fine = coarse
        fine_frequencies = coarse_frequencies
        fine_times = coarse_times
    else:
        fine, fine_frequencies, fine_times = _stft_magnitude(
            audio,
            sr=config.sr,
            hop_length=config.hop_length,
            n_fft=config.fine_n_fft,
            win_length=config.fine_win_length,
            fmin=config.analysis_fmin,
            fmax=config.analysis_fmax,
        )
    if coarse.shape[1] != fine.shape[1] or not np.allclose(coarse_times, fine_times):
        raise ValueError("coarse and fine STFT frame axes do not align")
    return coarse, coarse_frequencies, fine, fine_frequencies, fine_times


def select_multires_proposal(
    coarse_magnitude: np.ndarray,
    coarse_frequencies_hz: np.ndarray,
    candidates_hz: np.ndarray,
    fine_anchor_emission: np.ndarray,
    fine_shrp_emission: np.ndarray,
    config: HybridStftConfig,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Track long-window ridges, then rank them with short-window evidence."""
    anchor_z = robust_standardize_emission(fine_anchor_emission)
    shrp_z = robust_standardize_emission(fine_shrp_emission)
    proposals: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []

    for low, high in config.proposal_bands_hz:
        low = max(float(low), config.h1_min)
        high = min(float(high), config.h1_max)
        if high <= low:
            continue
        path = _coarse_peak_path(
            coarse_magnitude,
            coarse_frequencies_hz,
            (low, high),
            max_jump_hz=config.proposal_max_jump_hz,
        )
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
        raise ValueError("no valid coarse STFT H1 proposals")
    broad = (coarse_frequencies_hz >= config.h1_min) & (
        coarse_frequencies_hz <= config.h1_max
    )
    broad_frequencies = coarse_frequencies_hz[broad]
    broad_magnitude = coarse_magnitude[broad]
    broad_anchor_hz = float(
        broad_frequencies[int(np.argmax(np.mean(broad_magnitude, axis=1)))]
    )
    if broad_anchor_hz >= config.proposal_anchor_threshold_hz:
        selected = int(
            np.argmin(
                [abs(item["median_h1_hz"] - broad_anchor_hz) for item in diagnostics]
            )
        )
    else:
        selected = int(np.argmax([item["score"] for item in diagnostics]))
    return proposals[selected], diagnostics


def track_multires_stft_from_representations(
    coarse_magnitude: np.ndarray,
    coarse_frequencies_hz: np.ndarray,
    fine_magnitude: np.ndarray,
    fine_frequencies_hz: np.ndarray,
    times: np.ndarray,
    config: HybridStftConfig,
) -> HybridStftResult:
    """Track H1/F0 from precomputed aligned coarse and fine STFTs."""
    candidates, _, _, shrp, anchor = elelet_shrp_components(
        fine_magnitude,
        fine_frequencies_hz,
        config,
    )
    harmonic_power = elelet_harmonic_power_emission(
        fine_magnitude,
        fine_frequencies_hz,
        candidates,
        config,
    )
    standardized = robust_standardize_emission(harmonic_power)
    coarse_path, proposal_diagnostics = select_multires_proposal(
        coarse_magnitude,
        coarse_frequencies_hz,
        candidates,
        anchor,
        shrp,
        config,
    )
    allowed = np.abs(candidates[:, None] - coarse_path[None, :]) <= config.refinement_radius_hz
    constrained = np.where(allowed, standardized, -1e6)
    path_h1, _ = viterbi_smooth_path(
        candidates,
        constrained,
        smoothness=config.smoothness,
        transition_scale_hz=config.transition_scale_hz,
        max_jump_hz=config.max_jump_hz,
    )

    selected_proposal = int(
        np.argmin(
            [
                abs(item["median_h1_hz"] - float(np.median(coarse_path)))
                for item in proposal_diagnostics
            ]
        )
    )
    selected_score = float(proposal_diagnostics[selected_proposal]["score"])
    rivals = [
        float(item["score"])
        for index, item in enumerate(proposal_diagnostics)
        if index != selected_proposal
    ]
    proposal_margin = selected_score - max(rivals) if rivals else 12.0
    proposal_confidence = float(
        1.0 / (1.0 + np.exp(-np.clip(proposal_margin, -12.0, 12.0)))
    )
    confidence = path_confidence(candidates, standardized, path_h1) * proposal_confidence
    band_low = max(config.h1_min, float(np.min(coarse_path)) - config.refinement_radius_hz)
    band_high = min(config.h1_max, float(np.max(coarse_path)) + config.refinement_radius_hz)
    band_mask = (candidates >= band_low) & (candidates <= band_high)

    return HybridStftResult(
        time=np.asarray(times, dtype=np.float64)[: len(path_h1)],
        h1_hz=path_h1,
        f0_hz=0.5 * path_h1,
        confidence=confidence,
        proposal_confidence=proposal_confidence,
        selected_band_hz=(
            float(candidates[band_mask][0]),
            float(candidates[band_mask][-1]),
        ),
        coarse_path_h1_hz=coarse_path,
        candidates_h1_hz=candidates[band_mask],
        emission=constrained[band_mask],
    )


def track_hybrid_stft(
    audio: np.ndarray,
    config: HybridStftConfig = HybridStftConfig(),
) -> HybridStftResult:
    coarse, coarse_frequencies, fine, fine_frequencies, times = compute_multires_stft(
        audio,
        config,
    )
    return track_multires_stft_from_representations(
        coarse,
        coarse_frequencies,
        fine,
        fine_frequencies,
        times,
        config,
    )


def with_parameters(config: HybridStftConfig, **parameters) -> HybridStftConfig:
    return replace(config, **parameters)
