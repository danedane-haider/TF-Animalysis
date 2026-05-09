"""Shared naming helpers for F0/F1 extraction workflow outputs."""

from __future__ import annotations

from pathlib import Path


DEFAULT_ALGORITHM = "peak"


def normalize_algorithm(algorithm: str | None = None) -> str:
    value = (algorithm or DEFAULT_ALGORITHM).lower().strip().replace("-", "_")
    if not value:
        return DEFAULT_ALGORITHM
    return value


def extracted_dir_name(algorithm: str) -> str:
    return f"f0_{normalize_algorithm(algorithm)}"


def corrected_dir_name() -> str:
    return "f0_corrected"


def refined_dir_name() -> str:
    return "f0_refined"


def representation_dir_name(pipeline: str, fmax: float = 750.0) -> str:
    fmax_label = int(fmax) if float(fmax).is_integer() else str(fmax).replace(".", "p")
    return f"{pipeline.lower().strip()}_{fmax_label}"


def resolve_existing_dir(
    base_dir: Path,
    requested_name: str | None,
    preferred_name: str,
    legacy_name: str | None = None,
) -> Path:
    """Resolve a workflow directory, preferring the new name but accepting a legacy folder."""
    if requested_name:
        return base_dir / requested_name

    preferred = base_dir / preferred_name
    if preferred.exists():
        return preferred

    if legacy_name is not None:
        legacy = base_dir / legacy_name
        if legacy.exists():
            return legacy

    return preferred


# Backwards-compatible alias for code that still talks about the initial contour directory.
initial_dir_name = extracted_dir_name
