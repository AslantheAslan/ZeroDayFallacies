#!/usr/bin/env python3
"""Fast FAR-versus-DR plotting for leave-one-class-out OOD experiments.

Score direction
---------------
* MLP / HSAGE / HGAT / HGT:
  use the stored Mahalanobis/anomaly score directly. Larger = more anomalous.
* KumoRFM:
  anomaly_score = 1 - score_benign. Larger = more anomalous.

Performance optimizations
-------------------------
* Reads only required parquet columns.
* Uses ``roc_curve(..., drop_intermediate=True)``.
* Collapses duplicate FAR values in linear time after sorting.
* Avoids ``np.unique(scores)`` and its extra full sort.
* Interpolates and averages curves inside workers, returning fixed-size arrays.
* Groups all Kumo runs for one variant/class into one worker job.
* Limits nested BLAS/OpenMP/PyArrow-style thread pools to one thread per worker.
* Caches processed mean curves using file size + modification time signatures.
* Avoids ``bbox_inches='tight'`` for very large 500-dpi PNGs and uses low PNG
  compression by default for faster figure saving.

Outputs
-------
* <dataset>_far_dr_per_class.png
* <dataset>_far_dr_aggregated.png
* <dataset>_dr_at_far.csv
* <dataset>_dr_at_far_formatted.csv
"""

from __future__ import annotations

# Set these before importing NumPy/scikit-learn to avoid nested oversubscription.
import os

_INNER_THREADS = os.environ.get("FAR_DR_INNER_THREADS", "1")
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = _INNER_THREADS

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - depends on user environment
    pq = None

try:
    import fastparquet
except ImportError:  # pragma: no cover - depends on user environment
    fastparquet = None


MODEL_ORDER = ["MLP", "HSAGE", "HGAT", "HGT", "KumoRFM"]
VARIANT_ORDER = ["clean", "noisy"]
VARIANT_TITLES = {"clean": "Original features", "noisy": "Noisy features"}
KUMO_BENIGN_SCORE_COLUMN = "score_benign"
DEFAULT_NUM_WORKERS = 8
SCRIPT_VERSION = "6.1.0-publication"
CACHE_VERSION = 6
REQUIRED_DR_AT_FAR = 1e-4
DR_AT_FAR_DEFAULTS = (REQUIRED_DR_AT_FAR, 1e-3, 1e-2, 1e-1)

# FAR budgets for which attainability is diagnosed.  Fixed (not tied to
# --dr-at-far-values) so that cached curves remain comparable across runs.
DIAG_FAR_VALUES = (1e-4, 1e-3, 1e-2, 1e-1)

CLASS_DISPLAY_NAMES = {
    "dos": "DoS",
    "DoS": "DoS",
    "bruteForce": "bruteForce",
    "pingScan": "pingScan",
    "portScan": "portScan",
    "Bot": "Bot",
    "Brute_Force_-XSS": "XSS",
    "DoS_attacks-GoldenEye": "GoldenEye",
    "DoS_attacks-Hulk": "Hulk",
    "DoS_attacks-SlowHTTPTest": "SlowHTTPTest",
    "DoS_attacks-Slowloris": "Slowloris",
    "Brute_Force_-Web": "bruteForceWeb",
    "SQL_Injection": "SQL",
    "Infilteration": "Infiltration",
    "DDOS_attack-HOIC": "HOIC",
    "DDOS_attack-LOIC-UDP": "LOIC-UDP",
    "DDoS_attacks-LOIC-HTTP": "LOIC-HTTP",
    "SSH-Bruteforce": "bruteForceSSH",
    "FTP-BruteForce": "bruteForceFTP",
}

DATASET_CONFIGS = {
    "CIDDS": {
        "ood_classes": ["dos", "portScan", "pingScan", "bruteForce"],
        "zero_day_class": 4,
        "standard_results_dir": "results/in_RAID_paper/dataframes",
        # The clean CIDDS Kumo artifacts are stored once at the artifact root,
        # rather than inside run_<n> directories.
        "kumo_clean_template": (
            "{root}/kumo_cidds_loo/"
            "query_test__loo_{ood}/predictions_merged.parquet"
        ),
        "kumo_noisy_template": (
            "{root}/run_{run}/kumo_cidds_loo_noise/"
            "query_test__loo_{ood}__noise_m0_s0.05_seed42__flows/"
            "predictions_merged.parquet"
        ),
    },
    "NF-CSE-CIC-IDS2018-v3": {
        "standard_results_dir": "results/dataframes",
        "ood_classes": [
            "Bot",
            "Brute_Force_-Web",
            "Brute_Force_-XSS",
            "DDOS_attack-HOIC",
            "DDOS_attack-LOIC-UDP",
            "DDoS_attacks-LOIC-HTTP",
            "DoS_attacks-GoldenEye",
            "DoS_attacks-Hulk",
            "DoS_attacks-SlowHTTPTest",
            "DoS_attacks-Slowloris",
            "FTP-BruteForce",
            "Infilteration",
            "SQL_Injection",
            "SSH-Bruteforce",
        ],
        "zero_day_class": 14,
        "kumo_clean_template": (
            "{root}/run_{run}/kumo_rfm_loo/NF-CSE-CIC-IDS2018-v3/"
            "query_test__loo_{ood}/predictions_merged.parquet"
        ),
        "kumo_noisy_template": (
            "{root}/run_{run}/kumo_rfm_loo_noise/NF-CSE-CIC-IDS2018-v3/"
            "query_test__loo_{ood}__noise_m0_s0.05_seed42__flows/"
            "predictions_merged.parquet"
        ),
    },
}

TRUE_COLUMN_CANDIDATES = (
    "true_label",
    "y_true",
    "label",
    "target",
    "ground_truth",
)


@dataclass(frozen=True)
class MeanCurve:
    far: np.ndarray
    dr_mean: np.ndarray
    dr_std: np.ndarray
    n_runs: int
    n_attack_mean: float
    # --- attainability diagnostics, one entry per contributing run curve ---
    # min_far[i]        : smallest non-zero FAR any threshold of run i realises
    # n_points_le[i, k] : measured operating points with 0 < FAR <= DIAG_FAR_VALUES[k]
    # n_benign[i]       : benign samples in run i
    # n_distinct[i]     : distinct benign score values in run i
    min_far: np.ndarray
    n_points_le: np.ndarray
    n_benign: np.ndarray
    n_distinct: np.ndarray


MeanCurveStore = dict[str, dict[str, dict[str, MeanCurve]]]


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def normalize_model_name(raw_name: str) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "", raw_name.lower())
    return {
        "mlp": "MLP",
        "heterosage": "HSAGE",
        "heterographsage": "HSAGE",
        "hsage": "HSAGE",
        "heterogat": "HGAT",
        "heterographgat": "HGAT",
        "hgat": "HGAT",
        "hgt": "HGT",
        "kumorfm": "KumoRFM",
        "kumo": "KumoRFM",
    }.get(token)


def infer_variant(text: str) -> str:
    """Infer clean/noisy from the encoded noise magnitude in a filename.

    Standard-result filenames contain ``noise_0.0`` even for original data, so
    checking only for the word ``noise`` incorrectly labels every file noisy.
    """
    lowered = text.lower()
    magnitude_match = re.search(
        r"(?:^|[_-])noise[_-]?(\d+(?:\.\d+)?)", lowered
    )
    if magnitude_match:
        return "clean" if float(magnitude_match.group(1)) == 0.0 else "noisy"
    return "noisy" if re.search(r"(?:^|[_-])noisy(?:[_-]|$)", lowered) else "clean"


def parse_standard_filename(path: Path, dataset: str) -> tuple[str, str, str] | None:
    """Parse a standard-model result filename robustly.

    The previous implementation required the exact prefix
    ``benchmark_<dataset>_``.  That is unnecessarily brittle, especially for
    CIDDS result folders whose filenames may use a slightly different dataset
    token.  We instead identify the model from the ``_<model>_epochs`` section
    and the held-out class from the bracketed class token.
    """
    name = path.name

    model_match = re.search(
        r"(?:^|_)(MLP|HeteroSage|HSAGE|HeteroGAT|HGAT|HGT)_epochs(?:_|$)",
        name,
        flags=re.IGNORECASE,
    )
    if not model_match:
        return None
    model = normalize_model_name(model_match.group(1))
    if model is None or model == "KumoRFM":
        return None

    ood_match = re.search(r"\[['\"](.+?)['\"]\]", name)
    if not ood_match:
        return None

    # Dataset filtering is deliberately permissive.  When the filename embeds
    # the requested dataset, it is accepted; when it omits or abbreviates the
    # dataset name, the selected directory and held-out-class list still
    # provide the necessary filtering.
    del dataset
    return model, ood_match.group(1), infer_variant(name)


def parquet_columns(path: Path) -> list[str]:
    """Read parquet schema without materializing dataframe rows."""
    if pq is not None:
        return list(pq.ParquetFile(path).schema.names)
    if fastparquet is not None:
        return list(fastparquet.ParquetFile(path).columns)
    raise RuntimeError(
        "Selective parquet loading requires pyarrow or fastparquet. "
        "Install one with `pip install pyarrow` or `pip install fastparquet`."
    )


def read_selected_parquet(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only selected columns and disable Arrow's internal read threads."""
    if pq is not None:
        table = pq.read_table(path, columns=list(columns), use_threads=False)
        return table.to_pandas(use_threads=False, split_blocks=True)
    # fastparquet/pandas fallback; Joblib still limits common native threadpools.
    return pd.read_parquet(path, columns=list(columns))


def find_true_column_from_names(
    columns: Sequence[str], explicit: str | None = None
) -> str:
    if explicit:
        if explicit not in columns:
            raise KeyError(
                f"Requested true-label column '{explicit}' is absent. "
                f"Available columns: {list(columns)}"
            )
        return explicit
    for column in TRUE_COLUMN_CANDIDATES:
        if column in columns:
            return column
    raise KeyError(
        "Could not infer the true-label column. Use --kumo-true-column. "
        f"Available columns: {list(columns)}"
    )


def find_standard_anomaly_score_columns(
    columns: Sequence[str], explicit: Sequence[str] | None = None
) -> list[str]:
    """Follow results_all.py score-column selection and use values directly."""
    if explicit:
        missing = [column for column in explicit if column not in columns]
        if missing:
            raise KeyError(
                f"Requested standard anomaly-score columns are absent: {missing}. "
                f"Available columns: {list(columns)}"
            )
        return sorted(list(explicit), key=natural_key)

    if "score_1" in columns:
        score_columns = [c for c in columns if str(c).startswith("score")]
    else:
        score_columns = [
            c for c in columns if str(c).startswith("mahalanobis_score")
        ]
    return sorted(score_columns, key=natural_key)


def binary_mask_and_labels(
    labels: pd.Series,
    benign_class: int,
    zero_day_class: int,
    ood_label: str,
    benign_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row mask and binary labels without copying a dataframe subset."""
    if pd.api.types.is_numeric_dtype(labels):
        values = labels.to_numpy()
        mask = np.isin(values, [benign_class, zero_day_class])
        y_binary = (values[mask] == zero_day_class).astype(np.uint8)
        return mask, y_binary

    normalized = labels.astype(str).str.strip().str.casefold().to_numpy()
    ood_norm = str(ood_label).strip().casefold()
    benign_values = {"benign", "normal", str(benign_class).casefold()}
    if benign_label is not None:
        benign_values.add(str(benign_label).strip().casefold())

    benign_mask = np.isin(normalized, list(benign_values))
    ood_mask = normalized == ood_norm
    mask = benign_mask | ood_mask
    y_binary = (normalized[mask] == ood_norm).astype(np.uint8)
    return mask, y_binary


def collapse_roc_points(far: np.ndarray, dr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate FAR values using a vectorized grouped maximum."""
    far = np.asarray(far, dtype=np.float64)
    dr = np.asarray(dr, dtype=np.float64)
    order = np.argsort(far, kind="stable")
    far_sorted = far[order]
    dr_sorted = dr[order]

    starts = np.r_[0, np.flatnonzero(np.diff(far_sorted) != 0) + 1]
    unique_far = far_sorted[starts]
    max_dr = np.maximum.reduceat(dr_sorted, starts)

    if unique_far[0] > 0.0:
        unique_far = np.r_[0.0, unique_far]
        max_dr = np.r_[0.0, max_dr]
    if unique_far[-1] < 1.0:
        unique_far = np.r_[unique_far, 1.0]
        max_dr = np.r_[max_dr, 1.0]

    return unique_far, np.maximum.accumulate(max_dr)


def make_far_grid(dense_low_far: bool) -> np.ndarray:
    if dense_low_far:
        return np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, 1e-5, 101),
                    np.linspace(1e-5, 1e-4, 181),
                    np.linspace(1e-4, 1e-3, 451),
                    np.linspace(1e-3, 1e-2, 361),
                    np.linspace(1e-2, 1.0, 991),
                ]
            )
        )
    return np.linspace(0.0, 1.0, 1001)


def compute_interpolated_curve(
    y_binary: np.ndarray,
    scores: np.ndarray,
    far_grid: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    y_binary = np.asarray(y_binary, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    y_binary = y_binary[finite]
    scores = scores[finite]

    if y_binary.size == 0 or y_binary.min() == y_binary.max():
        return None
    if scores.size == 0 or scores.min() == scores.max():
        warnings.warn("Skipping a run because its score column is constant.")
        return None

    # drop_intermediate=False is required for the attainability diagnostic:
    # the dropped points are collinear, so the interpolated curve below is
    # numerically identical to the drop_intermediate=True version.
    far, dr, _ = roc_curve(
        y_binary,
        scores,
        pos_label=1,
        drop_intermediate=False,
    )

    positive_far = far[far > 0.0]
    diagnostics = {
        "min_far": float(positive_far.min()) if positive_far.size else np.inf,
        "n_points_le": np.asarray(
            [
                int(((far > 0.0) & (far <= value)).sum())
                for value in DIAG_FAR_VALUES
            ],
            dtype=np.int64,
        ),
        "n_benign": int((y_binary == 0).sum()),
        "n_distinct": int(np.unique(scores[y_binary == 0]).size),
    }

    far, dr = collapse_roc_points(far, dr)
    return np.interp(far_grid, far, dr), int(y_binary.sum()), diagnostics


def average_interpolated(
    interpolated: Sequence[np.ndarray],
    attack_counts: Sequence[int],
    far_grid: np.ndarray,
    diagnostics: Sequence[dict] = (),
) -> MeanCurve | None:
    if not interpolated:
        return None
    values = np.vstack(interpolated)
    diagnostics = list(diagnostics)
    if diagnostics:
        min_far = np.asarray([d["min_far"] for d in diagnostics], dtype=np.float64)
        n_points_le = np.vstack([d["n_points_le"] for d in diagnostics])
        n_benign = np.asarray([d["n_benign"] for d in diagnostics], dtype=np.int64)
        n_distinct = np.asarray([d["n_distinct"] for d in diagnostics], dtype=np.int64)
    else:  # pragma: no cover - defensive
        min_far = np.full(len(interpolated), np.nan)
        n_points_le = np.zeros((len(interpolated), len(DIAG_FAR_VALUES)), dtype=np.int64)
        n_benign = np.zeros(len(interpolated), dtype=np.int64)
        n_distinct = np.zeros(len(interpolated), dtype=np.int64)
    return MeanCurve(
        far=far_grid,
        dr_mean=values.mean(axis=0),
        dr_std=(
            values.std(axis=0, ddof=1)
            if len(interpolated) > 1
            else np.zeros_like(far_grid)
        ),
        n_runs=len(interpolated),
        n_attack_mean=float(np.mean(attack_counts)),
        min_far=min_far,
        n_points_le=n_points_le,
        n_benign=n_benign,
        n_distinct=n_distinct,
    )


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_path_for(cache_dir: Path, namespace: str, payload: dict) -> Path:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return cache_dir / namespace / f"{digest}.npz"


def load_curve_cache(path: Path) -> MeanCurve | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return MeanCurve(
                far=data["far"],
                dr_mean=data["dr_mean"],
                dr_std=data["dr_std"],
                n_runs=int(data["n_runs"].item()),
                n_attack_mean=float(data["n_attack_mean"].item()),
                min_far=data["min_far"],
                n_points_le=data["n_points_le"],
                n_benign=data["n_benign"],
                n_distinct=data["n_distinct"],
            )
    except Exception as exc:
        warnings.warn(f"Ignoring unreadable cache file {path}: {exc}")
        return None


def save_curve_cache(path: Path, curve: MeanCurve) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            far=curve.far,
            dr_mean=curve.dr_mean,
            dr_std=curve.dr_std,
            n_runs=np.asarray(curve.n_runs),
            n_attack_mean=np.asarray(curve.n_attack_mean),
            min_far=curve.min_far,
            n_points_le=curve.n_points_le,
            n_benign=curve.n_benign,
            n_distinct=curve.n_distinct,
        )
    os.replace(temporary, path)


def process_standard_file(
    path: Path,
    model: str,
    ood_class: str,
    variant: str,
    benign_class: int,
    zero_day_class: int,
    anomaly_score_columns: Sequence[str] | None,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
) -> tuple[str, str, str, MeanCurve | None, bool, str | None]:
    """Process all score/run columns in one standard parquet inside one worker."""
    try:
        columns = parquet_columns(path)
        if "true_label" not in columns:
            raise KeyError("Missing required 'true_label' column")
        score_columns = find_standard_anomaly_score_columns(
            columns, anomaly_score_columns
        )
        if not score_columns:
            raise KeyError(
                "No score* or mahalanobis_score* anomaly columns found. "
                "Use --standard-anomaly-score-columns to specify them."
            )

        # Use the same dense low-FAR grid as KumoRFM so DR is computed
        # per run at FAR values below 0.001 before runs are averaged.
        far_grid = make_far_grid(dense_low_far=True)
        payload = {
            "version": CACHE_VERSION,
            "kind": "standard",
            "file": file_signature(path),
            "model": model,
            "ood_class": ood_class,
            "variant": variant,
            "benign_class": benign_class,
            "zero_day_class": zero_day_class,
            "score_columns": score_columns,
            "grid_size": far_grid.size,
        }
        cache_path = cache_path_for(cache_dir, "standard", payload)
        if use_cache and not refresh_cache:
            cached = load_curve_cache(cache_path)
            if cached is not None:
                return variant, model, ood_class, cached, True, None

        required_columns = list(dict.fromkeys(["true_label", *score_columns]))
        with threadpool_limits(limits=1):
            df = read_selected_parquet(path, required_columns)
            mask, y_binary = binary_mask_and_labels(
                df["true_label"],
                benign_class=benign_class,
                zero_day_class=zero_day_class,
                ood_label=ood_class,
            )

            interpolated: list[np.ndarray] = []
            attack_counts: list[int] = []
            diagnostics: list[dict] = []
            for score_column in score_columns:
                result = compute_interpolated_curve(
                    y_binary=y_binary,
                    scores=df[score_column].to_numpy(dtype=np.float64)[mask],
                    far_grid=far_grid,
                )
                if result is not None:
                    curve_values, n_attack, diagnostic = result
                    interpolated.append(curve_values)
                    attack_counts.append(n_attack)
                    diagnostics.append(diagnostic)

        curve = average_interpolated(
            interpolated, attack_counts, far_grid, diagnostics
        )
        if curve is not None and use_cache:
            save_curve_cache(cache_path, curve)
        return variant, model, ood_class, curve, False, None
    except Exception as exc:
        return (
            variant,
            model,
            ood_class,
            None,
            False,
            f"[standard] Could not process {path}: {exc}",
        )


def render_kumo_path(
    template: str, root: Path, run: int, ood: str, dataset: str
) -> Path:
    return Path(template.format(root=str(root), run=run, ood=ood, dataset=dataset))


def kumo_path_candidates(
    template: str,
    root: Path,
    run: int,
    ood: str,
    dataset: str,
    variant: str,
) -> list[Path]:
    """Return supported Kumo locations, including legacy root-level clean data."""
    candidates = [render_kumo_path(template, root, run, ood, dataset)]
    if variant == "clean" and "/run_{run}/" in template:
        root_level_template = template.replace("/run_{run}/", "/")
        candidates.append(
            render_kumo_path(root_level_template, root, run, ood, dataset)
        )

    # Preserve order while removing duplicate paths. This is important for
    # root-level clean artifacts, which are independent of --runs.
    return list(dict.fromkeys(candidates))


def process_kumo_group(
    paths: Sequence[Path],
    variant: str,
    ood_class: str,
    benign_class: int,
    zero_day_class: int,
    true_column: str | None,
    benign_label: str | None,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
) -> tuple[str, str, MeanCurve | None, bool, str | None]:
    """Process all Kumo runs for one variant/held-out class in one worker."""
    try:
        far_grid = make_far_grid(dense_low_far=True)
        payload = {
            "version": CACHE_VERSION,
            "kind": "kumo",
            "files": [file_signature(path) for path in paths],
            "variant": variant,
            "ood_class": ood_class,
            "benign_class": benign_class,
            "zero_day_class": zero_day_class,
            "true_column": true_column,
            "benign_label": benign_label,
            "score_column": KUMO_BENIGN_SCORE_COLUMN,
            "grid_size": far_grid.size,
        }
        cache_path = cache_path_for(cache_dir, "kumo", payload)
        if use_cache and not refresh_cache:
            cached = load_curve_cache(cache_path)
            if cached is not None:
                return variant, ood_class, cached, True, None

        interpolated: list[np.ndarray] = []
        attack_counts: list[int] = []
        diagnostics: list[dict] = []
        with threadpool_limits(limits=1):
            for path in paths:
                columns = parquet_columns(path)
                true_col = find_true_column_from_names(columns, true_column)
                if KUMO_BENIGN_SCORE_COLUMN not in columns:
                    raise KeyError(
                        f"Missing required Kumo column "
                        f"'{KUMO_BENIGN_SCORE_COLUMN}' in {path}"
                    )

                df = read_selected_parquet(
                    path,
                    [true_col, KUMO_BENIGN_SCORE_COLUMN],
                )
                mask, y_binary = binary_mask_and_labels(
                    df[true_col],
                    benign_class=benign_class,
                    zero_day_class=zero_day_class,
                    ood_label=ood_class,
                    benign_label=benign_label,
                )
                benign_scores = df[KUMO_BENIGN_SCORE_COLUMN].to_numpy(
                    dtype=np.float64
                )[mask]
                finite = benign_scores[np.isfinite(benign_scores)]
                if finite.size and np.any((finite < 0.0) | (finite > 1.0)):
                    warnings.warn(
                        f"[{path}] score_benign values outside [0,1]; "
                        "still applying anomaly_score = 1 - score_benign."
                    )
                result = compute_interpolated_curve(
                    y_binary=y_binary,
                    scores=1.0 - benign_scores,
                    far_grid=far_grid,
                )
                if result is not None:
                    curve_values, n_attack, diagnostic = result
                    interpolated.append(curve_values)
                    attack_counts.append(n_attack)
                    diagnostics.append(diagnostic)

        curve = average_interpolated(
            interpolated, attack_counts, far_grid, diagnostics
        )
        if curve is not None and use_cache:
            save_curve_cache(cache_path, curve)
        return variant, ood_class, curve, False, None
    except Exception as exc:
        joined = ", ".join(str(path) for path in paths)
        return (
            variant,
            ood_class,
            None,
            False,
            f"[KumoRFM] Could not process group [{joined}]: {exc}",
        )


def add_mean_curve(
    store: MeanCurveStore,
    variant: str,
    model: str,
    ood_class: str,
    curve: MeanCurve | None,
) -> None:
    if curve is not None:
        store[variant][model][ood_class] = curve


def load_standard_mean_curves(
    store: MeanCurveStore,
    results_dir: Path,
    dataset: str,
    benign_class: int,
    zero_day_class: int,
    file_glob: str,
    requested_models: set[str],
    requested_ood_classes: set[str],
    anomaly_score_columns: Sequence[str] | None,
    strict: bool,
    num_workers: int,
    parallel_verbose: int,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
) -> None:
    files = sorted(results_dir.glob(file_glob))
    jobs: list[tuple[Path, str, str, str]] = []
    for path in files:
        parsed = parse_standard_filename(path, dataset)
        if parsed is None:
            continue
        model, ood_class, variant = parsed
        if model not in requested_models or ood_class not in requested_ood_classes:
            continue
        jobs.append((path, model, ood_class, variant))

    print(
        f"[standard] scanned {len(files)} parquet files; "
        f"parsed {len(jobs)} matching files; workers={num_workers}, "
        f"cache={'on' if use_cache else 'off'}"
    )
    if files and not jobs:
        examples = "\n  ".join(path.name for path in files[:5])
        warnings.warn(
            "No standard-result filenames matched the requested models and "
            "held-out classes. First filenames inspected:\n  " + examples
        )
    with parallel_config(
        backend="loky",
        n_jobs=num_workers,
        inner_max_num_threads=1,
    ):
        results = Parallel(verbose=parallel_verbose)(
            delayed(process_standard_file)(
                path=path,
                model=model,
                ood_class=ood_class,
                variant=variant,
                benign_class=benign_class,
                zero_day_class=zero_day_class,
                anomaly_score_columns=anomaly_score_columns,
                cache_dir=cache_dir,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
            )
            for path, model, ood_class, variant in jobs
        )

    cache_hits = 0
    for variant, model, ood_class, curve, cache_hit, error in results:
        cache_hits += int(cache_hit)
        if error:
            if strict:
                raise RuntimeError(error)
            warnings.warn(error)
            continue
        add_mean_curve(store, variant, model, ood_class, curve)
    print(f"[standard] completed; cache hits={cache_hits}/{len(jobs)}")


def load_kumo_mean_curves(
    store: MeanCurveStore,
    artifact_root: Path,
    dataset: str,
    ood_classes: Sequence[str],
    runs: Sequence[int],
    benign_class: int,
    zero_day_class: int,
    clean_template: str,
    noisy_template: str,
    true_column: str | None,
    benign_label: str | None,
    strict: bool,
    num_workers: int,
    parallel_verbose: int,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
) -> None:
    jobs: list[tuple[list[Path], str, str]] = []
    missing = 0
    for variant, template in (("clean", clean_template), ("noisy", noisy_template)):
        for ood_class in ood_classes:
            paths: list[Path] = []
            seen_paths: set[Path] = set()
            for run in runs:
                candidates = kumo_path_candidates(
                    template=template,
                    root=artifact_root,
                    run=run,
                    ood=ood_class,
                    dataset=dataset,
                    variant=variant,
                )
                existing = [path for path in candidates if path.exists()]
                if not existing:
                    missing += 1
                    continue
                for path in existing:
                    resolved = path.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        paths.append(path)
            if paths:
                jobs.append((paths, variant, ood_class))

    print(
        f"[KumoRFM] {len(jobs)} grouped class/variant jobs "
        f"({missing} requested run files missing); workers={num_workers}, "
        f"cache={'on' if use_cache else 'off'}"
    )
    with parallel_config(
        backend="loky",
        n_jobs=num_workers,
        inner_max_num_threads=1,
    ):
        results = Parallel(verbose=parallel_verbose)(
            delayed(process_kumo_group)(
                paths=paths,
                variant=variant,
                ood_class=ood_class,
                benign_class=benign_class,
                zero_day_class=zero_day_class,
                true_column=true_column,
                benign_label=benign_label,
                cache_dir=cache_dir,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
            )
            for paths, variant, ood_class in jobs
        )

    cache_hits = 0
    for variant, ood_class, curve, cache_hit, error in results:
        cache_hits += int(cache_hit)
        if error:
            if strict:
                raise RuntimeError(error)
            warnings.warn(error)
            continue
        add_mean_curve(store, variant, "KumoRFM", ood_class, curve)
    print(f"[KumoRFM] completed; cache hits={cache_hits}/{len(jobs)}")


def aggregate_class_curves(
    class_curves: Iterable[MeanCurve],
    dense_low_far: bool,
    aggregation: str,
) -> MeanCurve | None:
    """Aggregate held-out-class curves and retain both sources of deviation.

    The returned standard deviation combines:
    1. run-to-run variation within each held-out class, and
    2. variation between held-out-class mean curves.

    This is the law of total variance applied pointwise along the FAR grid.
    """
    curves = list(class_curves)
    if not curves:
        return None

    far_grid = make_far_grid(dense_low_far=dense_low_far)
    values = np.vstack(
        [np.interp(far_grid, curve.far, curve.dr_mean) for curve in curves]
    )
    within_variances = np.vstack(
        [
            np.square(np.interp(far_grid, curve.far, curve.dr_std))
            for curve in curves
        ]
    )

    if aggregation == "macro":
        weights = np.ones(len(curves), dtype=np.float64)
    elif aggregation == "attack-weighted":
        weights = np.asarray(
            [curve.n_attack_mean for curve in curves], dtype=np.float64
        )
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(curves), dtype=np.float64)
    weights /= weights.sum()

    mean = np.average(values, axis=0, weights=weights)
    between_variance = np.average(
        np.square(values - mean), axis=0, weights=weights
    )
    within_variance = np.average(
        within_variances, axis=0, weights=weights
    )
    total_variance = within_variance + between_variance

    return MeanCurve(
        far=far_grid,
        dr_mean=mean,
        dr_std=np.sqrt(np.maximum(total_variance, 0.0)),
        n_runs=sum(curve.n_runs for curve in curves),
        n_attack_mean=float(np.mean([curve.n_attack_mean for curve in curves])),
        min_far=np.concatenate([curve.min_far for curve in curves]),
        n_points_le=np.vstack([curve.n_points_le for curve in curves]),
        n_benign=np.concatenate([curve.n_benign for curve in curves]),
        n_distinct=np.concatenate([curve.n_distinct for curve in curves]),
    )


def get_aggregated_curve(
    mean_curves: MeanCurveStore,
    variant: str,
    model: str,
    aggregation: str,
) -> MeanCurve | None:
    class_map = mean_curves.get(variant, {}).get(model, {})
    return aggregate_class_curves(
        class_map.values(),
        dense_low_far=True,
        aggregation=aggregation,
    )

def setup_axis(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Alarm Rate (FAR)", fontsize=10)
    ax.set_ylabel("Detection Rate (DR)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.25)
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=0.8, color="0.55", zorder=0)


def display_ood_label(label: str) -> str:
    """Return the requested publication label for a raw held-out class name."""
    return CLASS_DISPLAY_NAMES.get(label, label)


def build_category_colors(
    labels: Sequence[str], cmap_name: str
) -> dict[str, tuple[float, float, float, float]]:
    """Return deterministic categorical colors for curves and legends.

    ``tab20`` is reordered to use its darker, more distinct colors before the
    paired lighter shades. This avoids assigning nearly identical neighboring
    colors to the first held-out classes.
    """
    cmap = plt.get_cmap(cmap_name)
    if cmap_name == "tab20":
        color_indices = list(range(0, cmap.N, 2)) + list(range(1, cmap.N, 2))
    else:
        color_indices = list(range(cmap.N))
    return {
        label: cmap(color_indices[index % len(color_indices)])
        for index, label in enumerate(labels)
    }


def add_bottom_legend(
    legend_ax: Axes,
    labels: Sequence[str],
    colors: dict[str, object],
    ncol: int,
    fontsize: float = 11.0,
) -> None:
    """Draw the same framed, centered legend style below either figure."""
    if not labels:
        return
    handles = [
        Line2D([0], [0], color=colors[label], linewidth=2.7)
        for label in labels
    ]
    legend_ax.legend(
        handles,
        list(labels),
        loc="center",
        ncol=max(1, min(ncol, len(labels))),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.72",
        fontsize=fontsize,
        handlelength=3.0,
        handletextpad=0.8,
        columnspacing=1.8,
        labelspacing=0.9,
        borderpad=0.9,
    )


def add_low_far_inset(
    ax: Axes,
    curves: Sequence[dict[str, object]],
    limit: float,
) -> None:
    """Add a low-FAR inset positioned above the main x-axis labels.

    Explicit tick labels are used instead of Matplotlib's scientific-notation
    offset text, preventing the inset's x-axis labels from colliding with the
    main plot's x-axis.
    """
    if not curves:
        return

    # [left, bottom, width, height] in parent-axis coordinates.  The raised
    # bottom position leaves a clear gap above the main x-axis labels.
    axins = ax.inset_axes([0.535, 0.205, 0.43, 0.365])

    for item in curves:
        far = np.asarray(item["far"])
        mean = np.asarray(item["mean"])
        color = item["color"]
        linewidth = float(item.get("linewidth", 1.2))
        axins.plot(far, mean, color=color, linewidth=linewidth)

        lower = item.get("lower")
        upper = item.get("upper")
        if lower is not None and upper is not None:
            axins.fill_between(
                far,
                np.asarray(lower),
                np.asarray(upper),
                color=color,
                alpha=float(item.get("fill_alpha", 0.10)),
                linewidth=0.0,
            )

    axins.set_xlim(0.0, limit)
    axins.set_ylim(0.0, 1.0)
    axins.set_title(f"FAR ≤ {limit:g}", fontsize=8, pad=3)
    axins.grid(True, alpha=0.2)
    axins.tick_params(axis="both", labelsize=7, pad=1.5)

    x_ticks = np.asarray([0.0, limit / 2.0, limit])
    axins.set_xticks(x_ticks)
    axins.set_xticklabels(
        ["0", f"{limit / 2.0:.0e}", f"{limit:.0e}"],
        fontsize=7,
    )
    axins.set_yticks([0.0, 0.5, 1.0])
    axins.xaxis.get_offset_text().set_visible(False)

def save_png(fig: plt.Figure, output_path: Path, dpi: int, compression: int) -> None:
    """Save as PNG and a matching PDF, then close the figure."""
    fig.savefig(
        output_path,
        dpi=dpi,
        format="png",
        pil_kwargs={"compress_level": compression},
    )
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, format="pdf")
    plt.close(fig)


def plot_per_class_figure(
    mean_curves: MeanCurveStore,
    models: Sequence[str],
    ood_classes: Sequence[str],
    dataset: str,
    output_path: Path,
    dpi: int,
    low_far_limit: float,
    show_std: bool,
    png_compression: int,
) -> None:
    del dataset  # Figure-level headlines are intentionally omitted.
    n_models = len(models)
    legend_columns = min(5, max(1, len(ood_classes)))
    legend_rows = int(np.ceil(len(ood_classes) / legend_columns))
    legend_ratio = max(0.18, 0.15 * legend_rows)
    figure_height = 9.8 + 0.70 * legend_rows

    fig = plt.figure(
        figsize=(5.1 * n_models, figure_height),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        nrows=3,
        ncols=n_models,
        height_ratios=[1.0, 1.0, legend_ratio],
    )
    axes = np.empty((2, n_models), dtype=object)
    for row in range(2):
        for col in range(n_models):
            axes[row, col] = fig.add_subplot(grid[row, col])
    legend_ax = fig.add_subplot(grid[2, :])
    legend_ax.axis("off")

    class_colors = build_category_colors(ood_classes, "tab20")
    plotted_classes: set[str] = set()

    for row, variant in enumerate(VARIANT_ORDER):
        for col, model in enumerate(models):
            ax = axes[row, col]
            setup_axis(ax, f"{model} — {VARIANT_TITLES[variant]}")
            inset_curves: list[dict[str, object]] = []
            available = mean_curves.get(variant, {}).get(model, {})

            for ood_class in ood_classes:
                curve = available.get(ood_class)
                if curve is None:
                    continue
                plotted_classes.add(ood_class)
                color = class_colors[ood_class]
                lower = np.clip(curve.dr_mean - curve.dr_std, 0.0, 1.0)
                upper = np.clip(curve.dr_mean + curve.dr_std, 0.0, 1.0)

                ax.plot(
                    curve.far,
                    curve.dr_mean,
                    linewidth=1.55,
                    color=color,
                )
                if show_std:
                    ax.fill_between(
                        curve.far,
                        lower,
                        upper,
                        color=color,
                        alpha=0.10,
                        linewidth=0.0,
                    )

                inset_curves.append(
                    {
                        "far": curve.far,
                        "mean": curve.dr_mean,
                        "lower": lower if show_std else None,
                        "upper": upper if show_std else None,
                        "color": color,
                        "linewidth": 1.1,
                        "fill_alpha": 0.09,
                    }
                )

            if not available:
                ax.text(
                    0.5,
                    0.5,
                    "No matching data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
            if inset_curves:
                add_low_far_inset(ax, inset_curves, low_far_limit)

    ordered_classes = [
        ood_class for ood_class in ood_classes if ood_class in plotted_classes
    ]
    add_bottom_legend(
        legend_ax,
        labels=[display_ood_label(name) for name in ordered_classes],
        colors={
            display_ood_label(name): class_colors[name]
            for name in ordered_classes
        },
        ncol=legend_columns,
        fontsize=10.5,
    )

    save_png(fig, output_path, dpi, png_compression)

def plot_aggregated_figure(
    mean_curves: MeanCurveStore,
    models: Sequence[str],
    dataset: str,
    output_path: Path,
    dpi: int,
    low_far_limit: float,
    aggregation: str,
    show_std: bool,
    png_compression: int,
) -> None:
    del dataset  # Figure-level headlines are intentionally omitted.

    fig = plt.figure(figsize=(14.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 0.16],
    )
    axes = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
        ],
        dtype=object,
    )
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")

    model_colors = build_category_colors(models, "tab10")
    plotted_models: set[str] = set()

    for ax, variant in zip(axes, VARIANT_ORDER):
        setup_axis(ax, VARIANT_TITLES[variant])
        inset_curves: list[dict[str, object]] = []
        plotted = 0

        for model in models:
            aggregate = get_aggregated_curve(
                mean_curves=mean_curves,
                variant=variant,
                model=model,
                aggregation=aggregation,
            )
            if aggregate is None:
                continue

            plotted += 1
            plotted_models.add(model)
            color = model_colors[model]
            lower = np.clip(aggregate.dr_mean - aggregate.dr_std, 0.0, 1.0)
            upper = np.clip(aggregate.dr_mean + aggregate.dr_std, 0.0, 1.0)

            ax.plot(
                aggregate.far,
                aggregate.dr_mean,
                linewidth=2.25,
                color=color,
            )
            if show_std:
                ax.fill_between(
                    aggregate.far,
                    lower,
                    upper,
                    color=color,
                    alpha=0.13,
                    linewidth=0.0,
                )

            inset_curves.append(
                {
                    "far": aggregate.far,
                    "mean": aggregate.dr_mean,
                    "lower": lower if show_std else None,
                    "upper": upper if show_std else None,
                    "color": color,
                    "linewidth": 1.45,
                    "fill_alpha": 0.11,
                }
            )

        if plotted:
            add_low_far_inset(ax, inset_curves, low_far_limit)
        else:
            ax.text(
                0.5,
                0.5,
                "No matching data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    ordered_models = [model for model in models if model in plotted_models]
    add_bottom_legend(
        legend_ax,
        labels=ordered_models,
        colors=model_colors,
        ncol=len(ordered_models),
        fontsize=11.5,
    )

    del aggregation
    save_png(fig, output_path, dpi, png_compression)


def format_far_value(value: float) -> str:
    """Compact, stable FAR label for CSV column names."""
    return f"{value:g}"


def build_dr_at_far_tables(
    mean_curves: MeanCurveStore,
    models: Sequence[str],
    aggregation: str,
    far_values: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build numeric long-form and publication-friendly wide DR@FAR tables."""
    raw_rows: list[dict[str, object]] = []
    formatted_rows: list[dict[str, object]] = []

    for model in models:
        formatted_row: dict[str, object] = {"Model": model}
        for variant in VARIANT_ORDER:
            aggregate = get_aggregated_curve(
                mean_curves=mean_curves,
                variant=variant,
                model=model,
                aggregation=aggregation,
            )
            class_count = len(mean_curves.get(variant, {}).get(model, {}))

            for far_value in far_values:
                if aggregate is None:
                    dr_mean = np.nan
                    dr_std = np.nan
                    run_count = 0
                else:
                    dr_mean = float(
                        np.interp(far_value, aggregate.far, aggregate.dr_mean)
                    )
                    dr_std = float(
                        np.interp(far_value, aggregate.far, aggregate.dr_std)
                    )
                    run_count = aggregate.n_runs

                raw_rows.append(
                    {
                        "Model": model,
                        "Variant": VARIANT_TITLES[variant],
                        "FAR": far_value,
                        "DR_mean": dr_mean,
                        "DR_std": dr_std,
                        "DR_mean_percent": 100.0 * dr_mean,
                        "DR_std_percent": 100.0 * dr_std,
                        "Held_out_classes": class_count,
                        "Run_curves": run_count,
                        "Aggregation": aggregation,
                    }
                )

                column = (
                    f"{VARIANT_TITLES[variant]} "
                    f"DR@FAR={format_far_value(far_value)} (%)"
                )
                if np.isfinite(dr_mean) and np.isfinite(dr_std):
                    formatted_row[column] = (
                        f"{100.0 * dr_mean:.2f} ± {100.0 * dr_std:.2f}"
                    )
                else:
                    formatted_row[column] = ""

        formatted_rows.append(formatted_row)

    return pd.DataFrame(raw_rows), pd.DataFrame(formatted_rows)


def save_dr_at_far_tables(
    mean_curves: MeanCurveStore,
    models: Sequence[str],
    aggregation: str,
    far_values: Sequence[float],
    raw_output_path: Path,
    formatted_output_path: Path,
) -> None:
    raw_df, formatted_df = build_dr_at_far_tables(
        mean_curves=mean_curves,
        models=models,
        aggregation=aggregation,
        far_values=far_values,
    )
    raw_df.to_csv(raw_output_path, index=False)
    formatted_df.to_csv(formatted_output_path, index=False)


def verdict_for(n_curves: int, n_attainable: int) -> str:
    """Decide whether a table cell needs the dagger footnote."""
    if n_curves == 0:
        return "no data"
    if n_attainable == 0:
        return "DAGGER  (interpolated: no curve attains this FAR)"
    if n_attainable < n_curves:
        return f"DAGGER* (only {n_attainable}/{n_curves} curves attain this FAR)"
    return "ok      (measured operating point)"


def build_attainability_tables(
    mean_curves: MeanCurveStore,
    models: Sequence[str],
    ood_classes: Sequence[str],
    aggregation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report, per table cell, whether the FAR budget is actually attainable.

    A DR@FAR entry is a genuine measurement only if at least one threshold of
    the underlying ROC curve realises a false alarm rate <= the budget.  Where
    no such threshold exists, ``np.interp`` returns a linear upper bound drawn
    from the origin to the first attainable operating point.
    """
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []

    for variant in VARIANT_ORDER:
        for model in models:
            class_map = mean_curves.get(variant, {}).get(model, {})

            for ood_class in ood_classes:
                curve = class_map.get(ood_class)
                if curve is None:
                    continue
                for index, far_value in enumerate(DIAG_FAR_VALUES):
                    attainable = int((curve.min_far <= far_value).sum())
                    per_class_rows.append(
                        {
                            "Model": model,
                            "Variant": VARIANT_TITLES[variant],
                            "OOD_class": display_ood_label(ood_class),
                            "FAR": far_value,
                            "Run_curves": int(curve.min_far.size),
                            "Curves_attaining_FAR": attainable,
                            "Min_attainable_FAR_median": float(
                                np.median(curve.min_far)
                            ),
                            "Points_below_FAR_median": float(
                                np.median(curve.n_points_le[:, index])
                            ),
                            "Benign_samples_median": float(
                                np.median(curve.n_benign)
                            ),
                            "Distinct_benign_scores_median": float(
                                np.median(curve.n_distinct)
                            ),
                            "Verdict": verdict_for(
                                int(curve.min_far.size), attainable
                            ),
                        }
                    )

            aggregate = get_aggregated_curve(
                mean_curves=mean_curves,
                variant=variant,
                model=model,
                aggregation=aggregation,
            )
            if aggregate is None:
                continue
            for index, far_value in enumerate(DIAG_FAR_VALUES):
                n_curves = int(aggregate.min_far.size)
                attainable = int((aggregate.min_far <= far_value).sum())
                summary_rows.append(
                    {
                        "Model": model,
                        "Variant": VARIANT_TITLES[variant],
                        "FAR": far_value,
                        "Held_out_classes": len(class_map),
                        "Run_curves": n_curves,
                        "Curves_attaining_FAR": attainable,
                        "Fraction_attaining": (
                            attainable / n_curves if n_curves else np.nan
                        ),
                        "Min_attainable_FAR_median": float(
                            np.median(aggregate.min_far)
                        ),
                        "Min_attainable_FAR_best": float(np.min(aggregate.min_far)),
                        "Points_below_FAR_median": float(
                            np.median(aggregate.n_points_le[:, index])
                        ),
                        "Benign_samples_median": float(
                            np.median(aggregate.n_benign)
                        ),
                        "Distinct_benign_scores_median": float(
                            np.median(aggregate.n_distinct)
                        ),
                        "Verdict": verdict_for(n_curves, attainable),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows)


def print_attainability_report(summary: pd.DataFrame) -> None:
    """Print the dagger decision for every DR@FAR table cell."""
    print("\n" + "=" * 108)
    print("DR@FAR ATTAINABILITY  --  does a threshold realising this FAR exist?")
    print("=" * 108)
    print(
        f"{'Variant':<18} {'Model':<9} {'FAR':>7} {'attain':>9} "
        f"{'min FAR':>11} {'pts<=FAR':>9} {'benign':>10} {'distinct':>9}  Verdict"
    )
    print("-" * 108)
    for variant_title in VARIANT_TITLES.values():
        block = summary[summary.Variant == variant_title]
        if block.empty:
            continue
        for _, row in block.iterrows():
            print(
                f"{row.Variant:<18} {row.Model:<9} {row.FAR:>7.0e} "
                f"{int(row.Curves_attaining_FAR):>4d}/{int(row.Run_curves):<4d} "
                f"{row.Min_attainable_FAR_median:>11.2e} "
                f"{row.Points_below_FAR_median:>9.0f} "
                f"{row.Benign_samples_median:>10.0f} "
                f"{row.Distinct_benign_scores_median:>9.0f}  {row.Verdict}"
            )
        print("-" * 108)
    print(
        "Cells marked DAGGER are linear upper bounds, not measured operating "
        "points.\nWhere a cell is 'ok' but DR(1e-3)/DR(1e-4) ~ 10, the ROC is "
        "genuinely linear in that\nregion (constant likelihood ratio); no "
        "dagger is needed, but no threshold there is\npreferable to any other."
    )
    print("=" * 108 + "\n")


def summarize_loaded_curves(
    mean_curves: MeanCurveStore, models: Sequence[str]
) -> None:
    print("\nLoaded curve summary")
    print("-" * 72)
    for variant in VARIANT_ORDER:
        for model in models:
            class_map = mean_curves.get(variant, {}).get(model, {})
            runs = sum(curve.n_runs for curve in class_map.values())
            print(
                f"{variant:5s} | {model:10s} | "
                f"classes={len(class_map):2d} | run-curves={runs:3d}"
            )


def resolve_dataset_settings(
    args: argparse.Namespace,
) -> tuple[list[str], int, str, str]:
    config = DATASET_CONFIGS.get(args.dataset, {})
    if args.ood_classes:
        ood_classes = list(args.ood_classes)
    elif "ood_classes" in config:
        ood_classes = list(config["ood_classes"])
    else:
        raise ValueError(
            f"Dataset '{args.dataset}' has no built-in OOD class list. "
            "Pass --ood-classes CLASS1 CLASS2 ..."
        )

    if args.zero_day_class is not None:
        zero_day_class = args.zero_day_class
    elif "zero_day_class" in config:
        zero_day_class = int(config["zero_day_class"])
    else:
        raise ValueError(
            f"Dataset '{args.dataset}' has no built-in zero-day class ID. "
            "Pass --zero-day-class ID."
        )

    clean_template = args.kumo_clean_template or config.get("kumo_clean_template")
    noisy_template = args.kumo_noisy_template or config.get("kumo_noisy_template")
    if not clean_template or not noisy_template:
        raise ValueError(
            f"Dataset '{args.dataset}' has no built-in Kumo path templates. "
            "Pass both --kumo-clean-template and --kumo-noisy-template."
        )
    return ood_classes, zero_day_class, str(clean_template), str(noisy_template)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast run-averaged FAR-versus-DR plotting."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--standard-results-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing non-Kumo parquet files. When omitted, "
            "CIDDS uses results/in_RAID_paper/dataframes and other built-in "
            "datasets use their configured default."
        ),
    )
    parser.add_argument("--standard-file-glob", default="*.parquet")
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts")
    )
    parser.add_argument(
        "--standard-anomaly-score-columns",
        "--standard-score-columns",
        dest="standard_anomaly_score_columns",
        nargs="+",
        default=None,
        help=(
            "Optional exact score columns for non-Kumo models. By default, "
            "score* is used when score_1 exists; otherwise "
            "mahalanobis_score* is used. Scores are not inverted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/far_dr_plots"),
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER
    )
    parser.add_argument("--ood-classes", nargs="+", default=None)
    parser.add_argument(
        "--runs", nargs="+", type=int, default=[0, 1, 2, 3, 4]
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Parallel worker processes (default: 8).",
    )
    parser.add_argument(
        "--parallel-verbose",
        type=int,
        default=5,
        help="Joblib progress verbosity (default: 5; use 0 to silence).",
    )
    parser.add_argument("--benign-class", type=int, default=0)
    parser.add_argument("--zero-day-class", type=int, default=None)
    parser.add_argument("--benign-label", default=None)
    parser.add_argument("--kumo-true-column", default=None)
    parser.add_argument("--kumo-clean-template", default=None)
    parser.add_argument("--kumo-noisy-template", default=None)
    parser.add_argument(
        "--aggregation",
        choices=["macro", "attack-weighted"],
        default="macro",
    )
    parser.add_argument("--low-far-limit", type=float, default=0.001)
    std_group = parser.add_mutually_exclusive_group()
    std_group.add_argument(
        "--show-std",
        dest="show_std",
        action="store_true",
        help="Show ±1 standard-deviation bands (default).",
    )
    std_group.add_argument(
        "--no-std",
        dest="show_std",
        action="store_false",
        help="Hide the standard-deviation bands.",
    )
    parser.set_defaults(show_std=True)
    parser.add_argument(
        "--dr-at-far-values",
        nargs="+",
        type=float,
        default=list(DR_AT_FAR_DEFAULTS),
        help=(
            "FAR operating points for the output comparison tables. "
            "FAR=0.0001 is always included "
            "(default: 0.0001 0.001 0.01 0.1)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        help="PNG compression 0-9. Lower is faster; default: 1.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/far_dr_cache"),
        help="Cache for precomputed mean curves.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing the curve cache.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute curves and overwrite matching cache entries.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[version] {SCRIPT_VERSION}")
    try:
        ood_classes, zero_day_class, clean_template, noisy_template = (
            resolve_dataset_settings(args)
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    dataset_config = DATASET_CONFIGS.get(args.dataset, {})
    standard_results_dir = args.standard_results_dir
    if standard_results_dir is None:
        standard_results_dir = Path(
            dataset_config.get("standard_results_dir", "results/dataframes")
        )
    print(f"[config] standard results directory: {standard_results_dir}")

    if args.num_workers < 1:
        print("--num-workers must be at least 1.", file=sys.stderr)
        return 2
    if not 0.0 < args.low_far_limit <= 1.0:
        print("--low-far-limit must be in (0, 1].", file=sys.stderr)
        return 2
    if any(not 0.0 <= value <= 1.0 for value in args.dr_at_far_values):
        print("--dr-at-far-values must all be in [0, 1].", file=sys.stderr)
        return 2
    # The publication tables must always report DR at FAR=0.0001.
    args.dr_at_far_values = sorted(
        {REQUIRED_DR_AT_FAR, *args.dr_at_far_values}
    )
    if not 0 <= args.png_compression <= 9:
        print("--png-compression must be between 0 and 9.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache
    if use_cache:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    store: MeanCurveStore = defaultdict(
        lambda: defaultdict(dict)
    )
    requested_models = set(args.models)
    requested_ood_classes = set(ood_classes)

    if requested_models - {"KumoRFM"}:
        load_standard_mean_curves(
            store=store,
            results_dir=standard_results_dir,
            dataset=args.dataset,
            benign_class=args.benign_class,
            zero_day_class=zero_day_class,
            file_glob=args.standard_file_glob,
            requested_models=requested_models,
            requested_ood_classes=requested_ood_classes,
            anomaly_score_columns=args.standard_anomaly_score_columns,
            strict=args.strict,
            num_workers=args.num_workers,
            parallel_verbose=args.parallel_verbose,
            cache_dir=args.cache_dir,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
        )

    if "KumoRFM" in requested_models:
        load_kumo_mean_curves(
            store=store,
            artifact_root=args.artifact_root,
            dataset=args.dataset,
            ood_classes=ood_classes,
            runs=args.runs,
            benign_class=args.benign_class,
            zero_day_class=zero_day_class,
            clean_template=clean_template,
            noisy_template=noisy_template,
            true_column=args.kumo_true_column,
            benign_label=args.benign_label,
            strict=args.strict,
            num_workers=args.num_workers,
            parallel_verbose=args.parallel_verbose,
            cache_dir=args.cache_dir,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
        )

    summarize_loaded_curves(store, args.models)
    if not any(
        store.get(variant, {}).get(model, {})
        for variant in VARIANT_ORDER
        for model in args.models
    ):
        print(
            "No usable curves were loaded. Check paths, dataset name, labels, "
            "and score columns.",
            file=sys.stderr,
        )
        return 1

    dataset_slug = slugify(args.dataset)
    per_class_path = args.output_dir / f"{dataset_slug}_far_dr_per_class.png"
    aggregate_path = args.output_dir / f"{dataset_slug}_far_dr_aggregated.png"
    dr_table_path = args.output_dir / f"{dataset_slug}_dr_at_far.csv"
    dr_table_formatted_path = (
        args.output_dir / f"{dataset_slug}_dr_at_far_formatted.csv"
    )
    attainability_path = args.output_dir / f"{dataset_slug}_dr_at_far_attainability.csv"
    attainability_per_class_path = (
        args.output_dir / f"{dataset_slug}_dr_at_far_attainability_per_class.csv"
    )

    print(f"[plot] Saving {per_class_path} at {args.dpi} dpi")
    plot_per_class_figure(
        mean_curves=store,
        models=args.models,
        ood_classes=ood_classes,
        dataset=args.dataset,
        output_path=per_class_path,
        dpi=args.dpi,
        low_far_limit=args.low_far_limit,
        show_std=args.show_std,
        png_compression=args.png_compression,
    )
    print(f"[plot] Saving {aggregate_path} at {args.dpi} dpi")
    plot_aggregated_figure(
        mean_curves=store,
        models=args.models,
        dataset=args.dataset,
        output_path=aggregate_path,
        dpi=args.dpi,
        low_far_limit=args.low_far_limit,
        aggregation=args.aggregation,
        show_std=args.show_std,
        png_compression=args.png_compression,
    )

    print(f"[table] Saving {dr_table_path}")
    print(f"[table] Saving {dr_table_formatted_path}")
    save_dr_at_far_tables(
        mean_curves=store,
        models=args.models,
        aggregation=args.aggregation,
        far_values=args.dr_at_far_values,
        raw_output_path=dr_table_path,
        formatted_output_path=dr_table_formatted_path,
    )

    attainability_df, attainability_per_class_df = build_attainability_tables(
        mean_curves=store,
        models=args.models,
        ood_classes=ood_classes,
        aggregation=args.aggregation,
    )
    print(f"[table] Saving {attainability_path}")
    attainability_df.to_csv(attainability_path, index=False)
    print(f"[table] Saving {attainability_per_class_path}")
    attainability_per_class_df.to_csv(attainability_per_class_path, index=False)
    print_attainability_report(attainability_df)

    print("\nSaved outputs:")
    print(f"  {per_class_path}")
    print(f"  {aggregate_path}")
    print(f"  {dr_table_path}")
    print(f"  {dr_table_formatted_path}")
    print(f"  {attainability_path}")
    print(f"  {attainability_per_class_path}")
    if use_cache:
        print(f"Curve cache: {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())