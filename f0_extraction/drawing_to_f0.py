"""Extract a synthesis-ready F0 contour from the printable drawing sheet."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image, ImageDraw
from scipy import ndimage


SUPPORTED_PHOTO_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class DrawingExtractionError(RuntimeError):
    """Raised when the plot or pen trace cannot be recovered reliably."""


@dataclass(frozen=True)
class DrawingExtractionConfig:
    f0_min_hz: float = 10.0
    f0_max_hz: float = 30.0
    frequency_scale: str = "linear"
    canonical_width: int = 1600
    canonical_height: int = 790
    max_detection_dimension: int = 2400
    minimum_plot_width_fraction: float = 0.45
    minimum_plot_height_fraction: float = 0.30
    minimum_contour_coverage: float = 0.55
    dark_threshold: float = 135.0

    def __post_init__(self) -> None:
        if self.f0_min_hz <= 0 or self.f0_max_hz <= self.f0_min_hz:
            raise ValueError("Require 0 < f0_min_hz < f0_max_hz")
        if self.frequency_scale not in {"linear", "log"}:
            raise ValueError("frequency_scale must be 'linear' or 'log'")
        if self.canonical_width < 200 or self.canonical_height < 100:
            raise ValueError("canonical dimensions are too small")


@dataclass
class DrawingContour:
    rectified_rgb: np.ndarray
    ink_mask: np.ndarray
    overlay_rgb: np.ndarray
    corners_xy: np.ndarray
    source_x_normalized: np.ndarray
    source_f0_hz: np.ndarray
    source_observed: np.ndarray


def _load_with_pillow(path: Path) -> Image.Image:
    with Image.open(path) as image:
        # Workshop photos arrive already upright. Some iPhone/sips conversions
        # retain a stale orientation tag even though the pixel array is correct,
        # so deliberately trust the pixels instead of applying EXIF rotation.
        return image.convert("RGB")


def load_photo(path: str | Path) -> Image.Image:
    """Load common image formats and use macOS ``sips`` as the HEIC fallback."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Photo not found: {path}")
    if path.suffix.lower() not in SUPPORTED_PHOTO_SUFFIXES:
        raise ValueError(f"Unsupported photo type {path.suffix!r}; expected one of {sorted(SUPPORTED_PHOTO_SUFFIXES)}")

    try:
        return _load_with_pillow(path)
    except Exception as pillow_error:
        if path.suffix.lower() not in {".heic", ".heif"}:
            raise DrawingExtractionError(f"Could not decode {path}: {pillow_error}") from pillow_error

    with tempfile.TemporaryDirectory(prefix="tf_animalysis_heic_") as temp_dir:
        converted = Path(temp_dir) / f"{path.stem}.png"
        try:
            completed = subprocess.run(
                ["sips", "-s", "format", "png", str(path), "--out", str(converted)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise DrawingExtractionError(
                "HEIC decoding needs macOS 'sips' or a Pillow HEIC plugin. Convert the photo to PNG/JPEG first."
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "unknown conversion error").strip()
            raise DrawingExtractionError(f"Could not convert HEIC photo with sips: {details}") from exc
        if not converted.exists():
            raise DrawingExtractionError(f"sips reported success but did not create {converted}: {completed.stdout}")
        return _load_with_pillow(converted)


def _resize_for_detection(image: Image.Image, max_dimension: int) -> Image.Image:
    scale = min(1.0, float(max_dimension) / max(image.size))
    if scale >= 1.0:
        return image.copy()
    size = tuple(max(1, int(round(value * scale))) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _blue_mask(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    chroma = maximum - minimum
    return (
        (blue >= 70)
        & (blue - red >= 20)
        & (blue - green >= 7)
        & (chroma >= 20)
    )


def _component_candidates(mask: np.ndarray) -> list[tuple[float, np.ndarray, tuple[int, int, int, int]]]:
    linked = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, count = ndimage.label(linked, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return []
    objects = ndimage.find_objects(labels)
    candidates: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    image_height, image_width = mask.shape
    for label_index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        width = int(x_slice.stop - x_slice.start)
        height = int(y_slice.stop - y_slice.start)
        width_fraction = width / image_width
        height_fraction = height / image_height
        if width_fraction < 0.10 or height_fraction < 0.04:
            continue
        component = labels == label_index
        area = int(np.count_nonzero(mask & component))
        score = float(area) * width_fraction * height_fraction**1.5
        candidates.append((score, component, (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)))
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def _robust_linear_fit(independent: np.ndarray, dependent: np.ndarray) -> tuple[float, float]:
    independent = np.asarray(independent, dtype=np.float64)
    dependent = np.asarray(dependent, dtype=np.float64)
    keep = np.isfinite(independent) & np.isfinite(dependent)
    if np.count_nonzero(keep) < 20:
        raise DrawingExtractionError("Too few blue-frame samples for edge fitting")
    for _ in range(6):
        slope, intercept = np.polyfit(independent[keep], dependent[keep], 1)
        residual = dependent - (slope * independent + intercept)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        threshold = max(2.5, 4.0 * 1.4826 * mad)
        updated = keep & (np.abs(residual - center) <= threshold)
        if np.count_nonzero(updated) < 20 or np.array_equal(updated, keep):
            break
        keep = updated
    slope, intercept = np.polyfit(independent[keep], dependent[keep], 1)
    return float(slope), float(intercept)


def _line_intersection(horizontal: tuple[float, float], vertical: tuple[float, float]) -> np.ndarray:
    """Intersect y = ax+b with x = cy+d."""
    a, b = horizontal
    c, d = vertical
    denominator = 1.0 - c * a
    if abs(denominator) < 1e-8:
        raise DrawingExtractionError("Detected blue frame has parallel/degenerate edges")
    x = (c * b + d) / denominator
    y = a * x + b
    return np.asarray([x, y], dtype=np.float64)


def _fit_plot_frame(mask: np.ndarray) -> np.ndarray:
    """Fit the four long outer frame lines, rejecting colored-paper outliers."""
    y_all, x_all = np.nonzero(mask)
    if x_all.size < 100:
        raise DrawingExtractionError("Too few blue-frame pixels were detected")
    x_low, x_high = np.percentile(x_all, [2.0, 98.0])
    y_low, y_high = np.percentile(y_all, [2.0, 98.0])
    span_x = max(1.0, x_high - x_low)
    span_y = max(1.0, y_high - y_low)
    x_min = max(0, int(np.floor(x_low - 0.03 * span_x)))
    x_max = min(mask.shape[1] - 1, int(np.ceil(x_high + 0.03 * span_x)))
    y_min = max(0, int(np.floor(y_low - 0.03 * span_y)))
    y_max = min(mask.shape[0] - 1, int(np.ceil(y_high + 0.03 * span_y)))

    horizontal_x: list[int] = []
    top_y: list[int] = []
    bottom_y: list[int] = []
    for x in range(x_min, x_max + 1):
        rows = np.flatnonzero(mask[y_min : y_max + 1, x])
        if rows.size:
            horizontal_x.append(x)
            top_y.append(int(rows[0] + y_min))
            bottom_y.append(int(rows[-1] + y_min))

    vertical_y: list[int] = []
    left_x: list[int] = []
    right_x: list[int] = []
    for y in range(y_min, y_max + 1):
        columns = np.flatnonzero(mask[y, x_min : x_max + 1])
        if columns.size:
            vertical_y.append(y)
            left_x.append(int(columns[0] + x_min))
            right_x.append(int(columns[-1] + x_min))

    top = _robust_linear_fit(np.asarray(horizontal_x), np.asarray(top_y))
    bottom = _robust_linear_fit(np.asarray(horizontal_x), np.asarray(bottom_y))
    left = _robust_linear_fit(np.asarray(vertical_y), np.asarray(left_x))
    right = _robust_linear_fit(np.asarray(vertical_y), np.asarray(right_x))
    return np.vstack(
        (
            _line_intersection(top, left),
            _line_intersection(top, right),
            _line_intersection(bottom, right),
            _line_intersection(bottom, left),
        )
    )


def detect_plot_corners(image: Image.Image, config: DrawingExtractionConfig) -> tuple[np.ndarray, Image.Image]:
    """Find the four corners of the connected blue plot grid."""
    detection_image = _resize_for_detection(image, config.max_detection_dimension)
    rgb = np.asarray(detection_image)
    mask = _blue_mask(rgb)
    candidates = _component_candidates(mask)
    image_width, image_height = detection_image.size

    selected: np.ndarray | None = None
    for _, component, bounds in candidates:
        x0, y0, x1, y1 = bounds
        if (
            (x1 - x0) / image_width >= config.minimum_plot_width_fraction
            and (y1 - y0) / image_height >= config.minimum_plot_height_fraction
        ):
            selected = component & mask
            break
    if selected is None:
        raise DrawingExtractionError(
            "Could not find the blue plot rectangle. Keep the full blue frame visible and print the template in color."
        )

    corners = _fit_plot_frame(selected)

    polygon_area = 0.5 * abs(
        np.dot(corners[:, 0], np.roll(corners[:, 1], -1))
        - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
    )
    if polygon_area < 0.08 * image_width * image_height:
        raise DrawingExtractionError("Detected blue frame is implausibly small or degenerate")
    return corners, detection_image


def rectify_plot(
    detection_image: Image.Image,
    corners_xy: np.ndarray,
    config: DrawingExtractionConfig,
) -> Image.Image:
    """Crop the upright blue frame and resize it; no rotation or homography."""
    left = max(0, int(np.floor(np.min(corners_xy[:, 0]))))
    top = max(0, int(np.floor(np.min(corners_xy[:, 1]))))
    right = min(detection_image.width, int(np.ceil(np.max(corners_xy[:, 0]))) + 1)
    bottom = min(detection_image.height, int(np.ceil(np.max(corners_xy[:, 1]))) + 1)
    if right - left < 100 or bottom - top < 100:
        raise DrawingExtractionError("Detected blue plot crop is too small")
    return detection_image.crop((left, top, right, bottom)).resize(
        (config.canonical_width, config.canonical_height),
        Image.Resampling.BICUBIC,
    )


def _neutral_dark_mask(rgb: np.ndarray, threshold: float) -> np.ndarray:
    values = rgb.astype(np.float32)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    # A cool shadow can tint genuinely black pen blue. Only reject blue-looking
    # pixels when they are bright enough to plausibly be printed grid/frame ink.
    blue_ink = (blue - red > 10.0) & (blue - green > 2.0) & (luminance >= 82.0)
    return (luminance < threshold) & ~blue_ink


def _select_pen_component(mask: np.ndarray, minimum_coverage: float) -> np.ndarray:
    structures = (
        ndimage.binary_closing(
            ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool)),
            structure=np.ones((3, 3), dtype=bool),
        ),
        ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool)),
    )
    height, width = mask.shape
    for cleaned in structures:
        labels, count = ndimage.label(cleaned, structure=np.ones((3, 3), dtype=bool))
        if count == 0:
            continue
        best_score = -1.0
        best_component: np.ndarray | None = None
        for label_index, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None:
                continue
            y_slice, x_slice = slices
            span_x = (x_slice.stop - x_slice.start) / width
            span_y = (y_slice.stop - y_slice.start) / height
            if span_x < minimum_coverage or span_y < 0.03:
                continue
            component = labels == label_index
            area = int(np.count_nonzero(component))
            score = span_x**2 * (0.25 + span_y) * area
            if score > best_score:
                best_score = score
                best_component = component
        if best_component is not None:
            return best_component
    raise DrawingExtractionError(
        "Could not find one continuous dark line across the plot. Use a thick black pen and avoid breaks or loops."
    )


def extract_contour(photo: str | Path | Image.Image, config: DrawingExtractionConfig | None = None) -> DrawingContour:
    """Rectify a photographed template and recover its centerline as F0."""
    config = config or DrawingExtractionConfig()
    image = load_photo(photo) if not isinstance(photo, Image.Image) else photo.convert("RGB")
    corners, detection_image = detect_plot_corners(image, config)
    rectified_image = rectify_plot(detection_image, corners, config)
    rectified_rgb = np.asarray(rectified_image)

    raw_ink = _neutral_dark_mask(rectified_rgb, config.dark_threshold)
    inset_x = max(4, int(round(config.canonical_width * 0.008)))
    inset_y = max(4, int(round(config.canonical_height * 0.008)))
    raw_ink[:inset_y, :] = False
    raw_ink[-inset_y:, :] = False
    raw_ink[:, :inset_x] = False
    raw_ink[:, -inset_x:] = False
    ink_mask = _select_pen_component(raw_ink, config.minimum_contour_coverage)

    x_coordinates = np.arange(config.canonical_width, dtype=np.float64)
    observed = np.zeros(config.canonical_width, dtype=bool)
    center_y = np.full(config.canonical_width, np.nan, dtype=np.float64)
    for x in range(config.canonical_width):
        rows = np.flatnonzero(ink_mask[:, x])
        if rows.size:
            observed[x] = True
            center_y[x] = float(np.median(rows))

    observed_x = x_coordinates[observed]
    if observed_x.size < 2:
        raise DrawingExtractionError("The detected pen trace has too few usable columns")
    coverage = (observed_x[-1] - observed_x[0] + 1.0) / config.canonical_width
    if coverage < config.minimum_contour_coverage:
        raise DrawingExtractionError(
            f"The pen trace covers only {coverage:.0%} of the time axis; expected at least {config.minimum_contour_coverage:.0%}."
        )

    center_y = np.interp(x_coordinates, observed_x, center_y[observed])
    center_y = ndimage.median_filter(center_y, size=5, mode="nearest")
    pitch_position = np.clip(1.0 - center_y / (config.canonical_height - 1.0), 0.0, 1.0)
    if config.frequency_scale == "log":
        f0_hz = config.f0_min_hz * (config.f0_max_hz / config.f0_min_hz) ** pitch_position
    else:
        f0_hz = config.f0_min_hz + pitch_position * (config.f0_max_hz - config.f0_min_hz)

    overlay = rectified_image.copy()
    draw = ImageDraw.Draw(overlay)
    overlay_points = [(int(x), int(round(y))) for x, y in zip(x_coordinates, center_y, strict=True)]
    draw.line(overlay_points, fill=(225, 35, 45), width=3)

    return DrawingContour(
        rectified_rgb=rectified_rgb,
        ink_mask=ink_mask,
        overlay_rgb=np.asarray(overlay),
        corners_xy=corners,
        source_x_normalized=x_coordinates / (config.canonical_width - 1.0),
        source_f0_hz=f0_hz.astype(np.float64),
        source_observed=observed,
    )


def audio_duration_seconds(path: str | Path) -> float:
    info = sf.info(str(Path(path).expanduser().resolve()))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Reference audio has invalid duration: {path}")
    return float(info.frames) / float(info.samplerate)


def interpolate_to_audio(
    contour: DrawingContour,
    duration_seconds: float,
    frame_resolution: float,
) -> pd.DataFrame:
    """Stretch the normalized drawing over the exact reference-audio duration."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if frame_resolution <= 0:
        raise ValueError("frame_resolution must be positive")
    times = np.arange(0.0, duration_seconds, frame_resolution, dtype=np.float64)
    if times.size < 2:
        times = np.asarray([0.0, max(0.0, duration_seconds - 1e-9)], dtype=np.float64)
    normalized_time = np.clip(times / duration_seconds, 0.0, 1.0)
    f0_hz = np.interp(normalized_time, contour.source_x_normalized, contour.source_f0_hz)

    source_observed = contour.source_observed.astype(np.float64)
    confidence = np.interp(normalized_time, contour.source_x_normalized, source_observed)
    confidence = np.clip(confidence, 0.0, 1.0)
    start_point = np.zeros(times.size, dtype=np.int8)
    end_point = np.zeros(times.size, dtype=np.int8)
    start_point[0] = 1
    end_point[-1] = 1
    return pd.DataFrame(
        {
            "time": times,
            # Elephant-Synth's current dataset loader expects H1 here and divides by two.
            "frequency": 2.0 * f0_hz,
            "f0_hz": f0_hz,
            "confidence": confidence,
            "start_point": start_point,
            "end_point": end_point,
            "algorithm": "photo_drawing",
            "frequency_role": "f1",
        }
    )


def save_diagnostics(contour: DrawingContour, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rectified": output_dir / "01_rectified_plot.png",
        "ink_mask": output_dir / "02_detected_ink.png",
        "overlay": output_dir / "03_extracted_contour_overlay.png",
    }
    Image.fromarray(contour.rectified_rgb).save(paths["rectified"])
    Image.fromarray((contour.ink_mask.astype(np.uint8) * 255), mode="L").save(paths["ink_mask"])
    Image.fromarray(contour.overlay_rgb).save(paths["overlay"])
    return paths
