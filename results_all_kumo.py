"""
results_all_kumo.py

Evaluates prediction results across all runs (1–5) and all OOD (zero-day) classes
for the Kumo CIDDS leave-one-out experiments.

For each OOD class, it aggregates metrics over runs and reports:
  - Mean and std of every metric
  - Separately for the non_zero_day and zero_day evaluation subsets

Usage:
    python results_all_kumo.py
    python results_all_kumo.py --artifact_root artifacts --runs 1 2 3 4 5 --noise
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd

from results import evaluate_single_run  # reuse your existing evaluation logic

warnings.filterwarnings("ignore")

dataset = "NF-CSE-CIC-IDS2018-v3"

# ── Config ────────────────────────────────────────────────────────────────────
if dataset == "CIDDS":
    OOD_CLASSES = ["dos", "portScan", "pingScan", "bruteForce"]

    # Path templates — two variants (clean / noise)
    PATH_CLEAN = (
        "{root}/run_{run}/kumo_cidds_loo/"
        "query_test__loo_{ood}/predictions_merged.parquet"
    )
    PATH_NOISE = (
        "{root}/run_{run}/kumo_cidds_loo_noise/"
        "query_test__loo_{ood}__noise_m0_s0.05_seed42__flows/"
        "predictions_merged.parquet"
    )

elif dataset == "NF-CSE-CIC-IDS2018-v3":
    OOD_CLASSES = [
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
    ]

    PATH_CLEAN = (
        "{root}/run_{run}/kumo_rfm_loo/NF-CSE-CIC-IDS2018-v3/"
        "query_test__loo_{ood}/predictions_merged.parquet"
    )
    PATH_NOISE = (
        "{root}/run_{run}/kumo_rfm_loo_noise/NF-CSE-CIC-IDS2018-v3/"
        "query_test__loo_{ood}__noise_m0_s0.05_seed42__flows/"
        "predictions_merged.parquet"
    )

else:
    raise ValueError("Dataset not recognized")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_path(template: str, root: str, run: int, ood: str) -> str:
    return template.format(root=root, run=run, ood=ood)


def aggregate_metrics(records: list[dict]) -> dict:
    """Given per-run metric dicts, return {metric: mean, metric_std: std}."""
    if not records:
        return {}
    df = pd.DataFrame(records)
    means = df.mean().rename(lambda c: c)
    stds = df.std().rename(lambda c: f"{c}_std")
    return {**means.to_dict(), **stds.to_dict()}


def format_summary(summary: dict) -> str:
    """Pretty-print a flat metric dict, pairing each mean with its std."""
    lines = []
    seen = set()
    for key, val in summary.items():
        if key.endswith("_std"):
            continue
        if key in seen:
            continue
        seen.add(key)
        std_key = f"{key}_std"
        std_val = summary.get(std_key, float("nan"))
        mean_str = f"{val:.4f}" if not pd.isna(val) else "NaN"
        std_str = f"{std_val:.4f}" if not pd.isna(std_val) else "NaN"
        lines.append(f"  {key:<35s}: {mean_str}  ± {std_str}")
    return "\n".join(lines)


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    artifact_root: str = "artifacts",
    runs: list[int] = None,
    use_noise: bool = False,
) -> dict:
    """
    Returns a nested dict:
        results[ood_class]["non_zero_day" | "zero_day"] = aggregated metric dict
    """
    if runs is None:
        runs = list(range(1, 6))

    path_template = PATH_NOISE if use_noise else PATH_CLEAN
    variant_label = "noise" if use_noise else "clean"

    all_results: dict[str, dict[str, dict]] = {}

    for ood in OOD_CLASSES:
        print(f"\n{'=' * 60}")
        print(f"  OOD class : {ood}  |  variant : {variant_label}")
        print(f"{'=' * 60}")

        per_run: dict[str, list] = {"non_zero_day": [], "zero_day": []}

        for run in runs:
            path = build_path(path_template, artifact_root, run, ood)

            if not os.path.exists(path):
                print(f"  [run {run}] ⚠  File not found, skipping: {path}")
                continue

            try:
                result = evaluate_single_run(
                    parquet_path=path,
                    dataset=dataset,
                    run=run,
                    is_kumo=True,
                    zero_day_class=14,
                    zero_day_label=ood,
                )
                for subset in ("non_zero_day", "zero_day"):
                    per_run[subset].append(result[subset])
                print(
                    f"  [run {run}] ✓  evaluated "
                    f"({len(per_run['non_zero_day'])} runs so far)"
                )
            except Exception as exc:
                print(f"  [run {run}] ✗  Error: {exc}")

        # Aggregate over runs
        aggregated = {
            subset: aggregate_metrics(records)
            for subset, records in per_run.items()
        }
        all_results[ood] = aggregated

        # Print summary for this OOD class
        for subset, metrics in aggregated.items():
            n_runs = len(per_run[subset])
            print(f"\n  ▶ {subset.upper()}  (n={n_runs} runs)")
            print(format_summary(metrics))

    return all_results


def results_to_dataframe(all_results: dict) -> dict[str, pd.DataFrame]:
    """
    Convert nested results into two DataFrames, one per evaluation subset.

    Each DataFrame is indexed by OOD class and contains both mean and _std
    columns. Raw numeric values are retained; formatting is applied later.
    """
    dfs = {}
    for subset in ("non_zero_day", "zero_day"):
        rows = {
            ood: all_results[ood][subset]
            for ood in OOD_CLASSES
            if ood in all_results
        }
        df = pd.DataFrame(rows).T
        dfs[subset] = df
    return dfs


def format_cell(mean_val, std_val) -> str:
    """Format a metric as 'avg ± std' in percent with one decimal place."""

    def fmt(value):
        if pd.isna(value):
            return "NaN"
        return f"{value * 100:.1f}"

    return f"{fmt(mean_val)} ± {fmt(std_val)}"


def build_formatted_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a raw aggregate DataFrame to display strings of the form
    'avg ± std'.
    """
    mean_cols = [column for column in df.columns if not column.endswith("_std")]
    formatted = {}
    for column in mean_cols:
        std_column = f"{column}_std"
        std_series = (
            df[std_column]
            if std_column in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        formatted[column] = [
            format_cell(mean, std)
            for mean, std in zip(df[column], std_series)
        ]
    return pd.DataFrame(formatted, index=df.index)


# ── LaTeX output ──────────────────────────────────────────────────────────────

# Only these metrics are printed in the LaTeX tables, in exactly this order.
# ``source`` is the metric key returned by evaluate_single_run; ``header`` is
# the text shown in the LaTeX table.
LATEX_METRICS = {
    "non_zero_day": [
        ("MACRO-F1", "Macro-F1"),
        ("AUROC", "AUROC"),
        ("AUPR-In", "AUPR In"),
        ("AUPR-Out", "AUPR Out"),
    ],
    "zero_day": [
        ("FalseAlarmRate (FAR)", "FalseAlarmRate (FAR)"),
        ("ZeroDay-DetectionRate (ZDDR)", "ZeroDay-DetectionRate (ZDDR)"),
        ("MACRO-F1", "Macro-F1"),
        ("AUROC", "AUROC"),
        ("AUPR-In", "AUPR In"),
        ("AUPR-Out", "AUPR Out"),
    ],
}


def latex_escape(value: object) -> str:
    """Escape text that will be placed in a LaTeX table cell or heading."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_label_slug(*parts: object) -> str:
    """Create a safe, stable LaTeX label from arbitrary text."""
    label = "-".join(str(part).strip().lower() for part in parts)
    label = re.sub(r"[^a-z0-9:.-]+", "-", label)
    return label.strip("-")


def normalize_metric_name(value: object) -> str:
    """Normalize capitalization and separators, without changing semantics."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_metric_column(columns, expected_name: str) -> str:
    """Find one requested evaluator metric column by spelling only."""
    expected = normalize_metric_name(expected_name)
    matches = [
        column
        for column in columns
        if normalize_metric_name(column) == expected
    ]

    if len(matches) == 1:
        return matches[0]

    available = ", ".join(str(column) for column in columns)
    if not matches:
        raise KeyError(
            f"Required LaTeX metric '{expected_name}' is missing. "
            f"Available metric columns: {available}"
        )

    raise KeyError(
        f"Required LaTeX metric '{expected_name}' is ambiguous: {matches}"
    )


def formatted_cell_to_latex(value: object) -> str:
    """Convert the existing CSV-style 'mean ± std' value to LaTeX."""
    return str(value).replace(" ± ", r" $\pm$ ")


def build_latex_df(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """
    Select only the requested LaTeX metrics and preserve their requested order.

    Values come from build_formatted_df(), the same formatting path used by the
    CSV output. No metric is recalculated or replaced with another metric.
    """
    if subset not in LATEX_METRICS:
        raise ValueError(f"Unknown evaluation subset: {subset}")

    formatted_df = build_formatted_df(df)
    latex_df = pd.DataFrame(index=formatted_df.index)

    for source_name, display_name in LATEX_METRICS[subset]:
        source_column = find_metric_column(formatted_df.columns, source_name)
        latex_df[latex_escape(display_name)] = formatted_df[source_column].map(
            formatted_cell_to_latex
        )

    latex_df.index = [latex_escape(index) for index in latex_df.index]
    return latex_df


def dataframe_to_latex(
    df: pd.DataFrame,
    subset: str,
    variant: str,
) -> str:
    """
    Render one aggregate DataFrame as a complete LaTeX table.

    The generated table uses booktabs, so include this in the LaTeX preamble:
        \\usepackage{booktabs}
    """
    latex_df = build_latex_df(df, subset=subset)
    subset_label = subset.replace("_", " ")

    column_spec = "l" + "c" * len(latex_df.columns)
    header = "OOD class & " + " & ".join(latex_df.columns) + r" \\"

    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        (
            rf"\caption{{Kumo {latex_escape(variant)} results on the "
            rf"{latex_escape(dataset)} dataset for the "
            rf"{latex_escape(subset_label)} subset. Values are mean "
            rf"$\pm$ standard deviation (\%).}}"
        ),
        rf"\label{{tab:{latex_label_slug('kumo', variant, subset)}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for ood_class, row in latex_df.iterrows():
        values = [ood_class] + [str(row[column]) for column in latex_df.columns]
        lines.append(" & ".join(values) + r" \\ ")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def print_latex_tables(
    dfs: dict[str, pd.DataFrame],
    variant: str,
) -> None:
    """Print one complete, explicitly filtered LaTeX table per subset."""
    print(f"\n\n{'#' * 60}")
    print("  LATEX TABLES")
    print(f"{'#' * 60}")

    for subset in ("non_zero_day", "zero_day"):
        print(f"\n% ── {subset.upper()} ──")
        print(dataframe_to_latex(dfs[subset], subset=subset, variant=variant))


# ── CSV output ────────────────────────────────────────────────────────────────

def save_results(
    dfs: dict[str, pd.DataFrame],
    out_dir: str,
    variant: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for subset, df in dfs.items():
        formatted_df = build_formatted_df(df)
        csv_path = os.path.join(out_dir, f"kumo_{variant}_{subset}.csv")
        formatted_df.to_csv(csv_path)
        print(f"\n💾  Saved: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Kumo evaluation metrics over runs."
    )
    parser.add_argument(
        "--artifact_root",
        default="artifacts",
        help="Root directory for artifact files (default: artifacts)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=list(range(0, 5)),
        help="Run indices to include (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Use noise variant of the parquet paths",
    )
    parser.add_argument(
        "--out_dir",
        default="results/kumo_summary",
        help="Directory to save CSV summaries (default: results/kumo_summary)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Skip saving CSVs",
    )
    parser.add_argument(
        "--no_latex",
        action="store_true",
        help="Skip printing the final LaTeX tables",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n🚀  Kumo evaluation — runs {args.runs}, noise={args.noise}")

    all_results = run_evaluation(
        artifact_root=args.artifact_root,
        runs=args.runs,
        use_noise=args.noise,
    )

    dfs = results_to_dataframe(all_results)
    variant = "noise" if args.noise else "clean"

    # Print final consolidated tables
    print(f"\n\n{'#' * 60}")
    print("  FINAL CONSOLIDATED TABLES")
    print(f"{'#' * 60}")
    for subset, df in dfs.items():
        print(f"\n── {subset.upper()} ──")
        # Show only mean columns for readability; std columns remain in the CSV.
        mean_cols = [column for column in df.columns if not column.endswith("_std")]
        print(df[mean_cols].to_string())

    # New feature: print complete LaTeX tables for both evaluation subsets.
    if not args.no_latex:
        print_latex_tables(dfs, variant=variant)

    if not args.no_save:
        save_results(dfs, args.out_dir, variant)
