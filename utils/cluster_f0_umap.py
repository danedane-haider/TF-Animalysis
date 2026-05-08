#!/usr/bin/env python3
"""Cluster F0 contours from CSV files using UMAP embeddings."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class ContourRecord:
    file_path: Path
    file_name: str
    frequency: np.ndarray
    time: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster F0 contour CSVs using UMAP + KMeans."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("rumbles/test/f0_corrected_refined"),
        help="Directory containing input .csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rumbles/test/f0_corrected_refined/umap_clustering"),
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--resample-length",
        type=int,
        default=128,
        help="Number of points per contour after interpolation.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=6,
        help="Number of clusters for KMeans in UMAP space.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["balanced", "frequency_main", "modulation_main"],
        default="balanced",
        help=(
            "Feature emphasis mode: "
            "balanced (pitch shape + summary), "
            "frequency_main (absolute frequency contour), "
            "modulation_main (change/derivative contour)."
        ),
    )
    return parser.parse_args()


def load_contour(path: Path) -> ContourRecord | None:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        return None
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    if "frequency" not in data.dtype.names:
        return None

    # Input tracks are first-harmonic values; divide by 2 to estimate F0.
    frequency = np.asarray(data["frequency"], dtype=float).reshape(-1) / 2.0
    if "time" in data.dtype.names:
        time = np.asarray(data["time"], dtype=float).reshape(-1)
    else:
        time = np.arange(frequency.size, dtype=float)

    mask = np.isfinite(frequency) & np.isfinite(time) & (frequency > 0.0)
    frequency = frequency[mask]
    time = time[mask]
    if frequency.size < 2:
        return None

    if np.any(np.diff(time) < 0):
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        frequency = frequency[sort_idx]

    return ContourRecord(file_path=path, file_name=path.name, frequency=frequency, time=time)


def load_all_contours(input_dir: Path) -> list[ContourRecord]:
    records: list[ContourRecord] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        rec = load_contour(csv_path)
        if rec is not None:
            records.append(rec)
    return records


def resample_contour(time: np.ndarray, frequency: np.ndarray, target_len: int) -> np.ndarray:
    if target_len < 2:
        raise ValueError("target_len must be >= 2")

    t0 = float(time[0])
    t1 = float(time[-1])
    if t1 <= t0:
        source_x = np.linspace(0.0, 1.0, frequency.size, dtype=float)
    else:
        source_x = (time - t0) / (t1 - t0)

    target_x = np.linspace(0.0, 1.0, target_len, dtype=float)
    return np.interp(target_x, source_x, frequency)


def zscore(vec: np.ndarray) -> np.ndarray:
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    if std == 0.0:
        return np.zeros_like(vec)
    return (vec - mean) / std


def contour_features(
    record: ContourRecord,
    resample_len: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    resampled = resample_contour(record.time, record.frequency, resample_len)
    safe_resampled = np.clip(resampled, 1e-6, None)
    log_contour = np.log2(safe_resampled)
    shape = zscore(log_contour)
    modulation = np.diff(shape, prepend=shape[0])
    curvature = np.diff(modulation, prepend=modulation[0])

    mean_freq = float(np.mean(record.frequency))
    std_freq = float(np.std(record.frequency))
    duration = float(record.time[-1] - record.time[0])

    if feature_mode == "frequency_main":
        summary = np.array([mean_freq, std_freq, duration], dtype=float)
        # Keep absolute contour dominant; include only a weak motion cue.
        features = np.concatenate([log_contour, 0.25 * modulation, summary], dtype=float)
    elif feature_mode == "modulation_main":
        weak_level = np.array([0.2 * mean_freq, 0.5 * std_freq, duration], dtype=float)
        # Focus on movement patterns, then curvature for modulation detail.
        features = np.concatenate([modulation, 0.6 * curvature, weak_level], dtype=float)
    else:
        summary = np.array([mean_freq, std_freq, duration], dtype=float)
        features = np.concatenate([shape, summary], dtype=float)

    stats = {
        "n_points": float(record.frequency.size),
        "duration_sec": duration,
        "mean_frequency": mean_freq,
        "std_frequency": std_freq,
    }
    return features, resampled, stats


def write_assignments_csv(
    output_path: Path,
    records: Iterable[ContourRecord],
    labels: np.ndarray,
    embedding: np.ndarray,
    stats: list[dict[str, float]],
) -> None:
    fieldnames = [
        "file_name",
        "file_path",
        "cluster",
        "umap_x",
        "umap_y",
        "n_points",
        "duration_sec",
        "mean_frequency",
        "std_frequency",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, rec in enumerate(records):
            row = {
                "file_name": rec.file_name,
                "file_path": str(rec.file_path),
                "cluster": int(labels[idx]),
                "umap_x": float(embedding[idx, 0]),
                "umap_y": float(embedding[idx, 1]),
                "n_points": int(stats[idx]["n_points"]),
                "duration_sec": float(stats[idx]["duration_sec"]),
                "mean_frequency": float(stats[idx]["mean_frequency"]),
                "std_frequency": float(stats[idx]["std_frequency"]),
            }
            writer.writerow(row)


def plot_umap_scatter(output_path: Path, embedding: np.ndarray, labels: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = sorted(np.unique(labels))
    plt.figure(figsize=(9, 7), dpi=140)
    cmap = plt.get_cmap("tab10")
    for i, cluster_id in enumerate(uniq):
        idx = labels == cluster_id
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=34,
            alpha=0.9,
            color=cmap(i % 10),
            label=f"Cluster {cluster_id}",
        )
    plt.title("UMAP Clusters of F0 Contours")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cluster_contours(
    output_path: Path,
    contours: np.ndarray,
    labels: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = sorted(np.unique(labels))
    x = np.linspace(0.0, 1.0, contours.shape[1], dtype=float)
    plt.figure(figsize=(10, 6), dpi=140)
    cmap = plt.get_cmap("tab10")
    for i, cluster_id in enumerate(uniq):
        idx = labels == cluster_id
        cluster_contours = contours[idx]
        mean_shape = np.mean(cluster_contours, axis=0)
        std_shape = np.std(cluster_contours, axis=0)
        color = cmap(i % 10)
        plt.plot(
            x,
            mean_shape,
            linewidth=2.0,
            color=color,
            label=f"Cluster {cluster_id} (n={int(np.sum(idx))})",
        )
        plt.fill_between(
            x,
            mean_shape - std_shape,
            mean_shape + std_shape,
            color=color,
            alpha=0.2,
            linewidth=0.0,
        )
    plt.title("Cluster Mean Contours with ±1 SD (raw frequency)")
    plt.xlabel("Normalized Time")
    plt.ylabel("Frequency")
    plt.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_contours(input_dir)
    if len(records) < 3:
        raise RuntimeError(
            f"Need at least 3 usable CSVs for clustering, found {len(records)} in {input_dir}."
        )

    feature_rows: list[np.ndarray] = []
    contour_shapes: list[np.ndarray] = []
    stats: list[dict[str, float]] = []
    for rec in records:
        feat, plot_shape, st = contour_features(rec, args.resample_length, args.feature_mode)
        feature_rows.append(feat)
        contour_shapes.append(plot_shape)
        stats.append(st)

    x = np.vstack(feature_rows)
    shape_matrix = np.vstack(contour_shapes)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    try:
        import umap.umap_ as umap
    except ImportError as exc:
        raise RuntimeError(
            "umap-learn is not installed. Install it with: python3 -m pip install umap-learn"
        ) from exc

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    n_samples = x_scaled.shape[0]
    n_neighbors = min(max(2, args.n_neighbors), n_samples - 1)
    n_clusters = min(max(2, args.n_clusters), n_samples - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(x_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=args.random_state, n_init="auto")
    labels = kmeans.fit_predict(embedding)
    sil = silhouette_score(embedding, labels)

    assignments_path = output_dir / "f0_umap_cluster_assignments.csv"
    write_assignments_csv(assignments_path, records, labels, embedding, stats)

    scatter_path = output_dir / "f0_umap_scatter.png"
    plot_umap_scatter(scatter_path, embedding, labels)

    contour_plot_path = output_dir / "f0_cluster_mean_contours.png"
    plot_cluster_contours(contour_plot_path, shape_matrix, labels)

    np.save(output_dir / "f0_umap_embedding.npy", embedding)
    np.save(output_dir / "f0_umap_labels.npy", labels)

    print(f"Input directory: {input_dir}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Contours clustered: {len(records)}")
    print(f"UMAP n_neighbors: {n_neighbors}")
    print(f"KMeans n_clusters: {n_clusters}")
    print(f"Silhouette score (UMAP space): {sil:.4f}")
    print(f"Assignments CSV: {assignments_path}")
    print(f"Scatter plot: {scatter_path}")
    print(f"Mean contour plot: {contour_plot_path}")


if __name__ == "__main__":
    main()
