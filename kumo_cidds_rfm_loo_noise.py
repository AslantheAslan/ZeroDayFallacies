#!/usr/bin/env python3
"""
KumoRFM leave-one-out pipeline for CIDDS with optional Gaussian noise augmentation.

This version keeps the working leave-one-out KumoRFM setup and fixes the practical
issues seen with the first noise-enabled variant:

- Gaussian noise is applied with compact float32 storage to avoid dtype inflation.
- By default, noise is applied only to FLOW feature columns, which is the closest
  useful analogue to the original augmentation while keeping Kumo context sizes smaller.
- Host-feature noise can still be enabled explicitly.
- Numeric feature columns are downcast to float32 even without noise to keep the Kumo
  payload compact.
- Optional automatic batch-size backoff handles Kumo's 30MB context limit more robustly.

Fair protocol preserved:
- context examples come only from df_train.parquet
- query rows come from df_val.parquet or df_test.parquet
- query targets are masked before Kumo inference
- leave-one-out removes a named class from df_train before graph creation
- --unknown-only can restrict evaluation to only the left-out class in the query split
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


KNOWN_CIDDS_LABELS = ["benign", "bruteForce", "dos", "pingScan", "portScan"]
DEFAULT_DATA_DIR = "./data/CIDDS"
DEFAULT_KUMO_URL = "https://kumorfm.ai/api"
KUMO_CONTEXT_LIMIT_SNIPPET = "Context size exceeds the 30MB limit"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def _sanitize_identifier(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"c_{name}"
    return name


def _build_unique_name_map(columns: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for col in columns:
        base = _sanitize_identifier(col)
        if base not in used:
            used[base] = 0
            out[col] = base
        else:
            used[base] += 1
            out[col] = f"{base}_{used[base]}"
    return out


def _canonicalize_ip(value: object) -> str:
    text = str(value).strip()
    return text if text else "0.0.0.0"


def _pql_literal(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _to_prediction_df(pred_obj) -> pd.DataFrame:
    if isinstance(pred_obj, pd.DataFrame):
        return pred_obj.copy()
    if isinstance(pred_obj, pd.Series):
        return pred_obj.to_frame(name="prediction")
    if isinstance(pred_obj, list):
        return pd.DataFrame({"prediction": pred_obj})
    try:
        return pd.DataFrame(pred_obj)
    except Exception as e:
        raise TypeError(f"Unsupported Kumo prediction object type: {type(pred_obj)}") from e


def _downcast_numeric_frame(df: pd.DataFrame, exclude: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    exclude_set = set(exclude)
    for col in out.columns:
        if col in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    return out


# -----------------------------------------------------------------------------
# Split loading and label handling
# -----------------------------------------------------------------------------


def detect_label_columns(df: pd.DataFrame) -> List[str]:
    labels = [c for c in KNOWN_CIDDS_LABELS if c in df.columns]
    if labels:
        return labels

    numeric_binary: List[str] = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        unique_vals = set(pd.unique(series))
        if unique_vals.issubset({0, 1, False, True}):
            numeric_binary.append(col)

    if len(numeric_binary) >= 2:
        return numeric_binary

    raise ValueError(
        "Could not infer CIDDS one-hot label columns. Expected columns like "
        f"{KNOWN_CIDDS_LABELS} or equivalent one-hot columns."
    )



def one_hot_to_attack_class(df: pd.DataFrame, label_cols: Sequence[str]) -> pd.Series:
    arr = df[list(label_cols)].to_numpy()
    row_sums = np.nansum(arr, axis=1)
    bad_idx = np.where(row_sums != 1)[0]
    if len(bad_idx) > 0:
        preview = bad_idx[:10].tolist()
        raise ValueError(
            "Expected exactly one active attack label per row, but found invalid one-hot rows. "
            f"Example row indices: {preview}"
        )
    label_idx = np.argmax(arr, axis=1)
    return pd.Series([label_cols[i] for i in label_idx], index=df.index, dtype="string")


@dataclass
class CIDDSSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_cols: List[str]



def load_cidds_splits(data_dir: Path) -> CIDDSSplits:
    train = _safe_read_parquet(data_dir / "df_train.parquet")
    val = _safe_read_parquet(data_dir / "df_val.parquet")
    test = _safe_read_parquet(data_dir / "df_test.parquet")
    label_cols = detect_label_columns(train)
    return CIDDSSplits(train=train, val=val, test=test, label_cols=label_cols)


# -----------------------------------------------------------------------------
# Filtering for leave-one-out
# -----------------------------------------------------------------------------


def filter_train_leave_one_out(
    train_df: pd.DataFrame,
    label_cols: Sequence[str],
    left_out_label: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_attack = one_hot_to_attack_class(train_df, label_cols)
    stats: Dict[str, object] = {
        "train_rows_before": int(len(train_df)),
        "left_out_label": left_out_label,
        "n_left_out_removed": 0,
        "train_rows_after": int(len(train_df)),
    }

    if left_out_label is None:
        return train_df.copy().reset_index(drop=True), stats

    if left_out_label not in label_cols:
        raise ValueError(
            f"--left-out-label={left_out_label!r} is not among detected labels: {list(label_cols)}"
        )

    keep_mask = train_attack != left_out_label
    filtered = train_df.loc[keep_mask].copy().reset_index(drop=True)
    removed = int((~keep_mask).sum())

    stats["n_left_out_removed"] = removed
    stats["train_rows_after"] = int(len(filtered))
    return filtered, stats



def filter_query_unknown_only(
    query_df: pd.DataFrame,
    label_cols: Sequence[str],
    left_out_label: Optional[str],
    unknown_only: bool,
    max_query_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    stats: Dict[str, object] = {
        "query_rows_before": int(len(query_df)),
        "unknown_only": bool(unknown_only),
        "query_rows_after": int(len(query_df)),
        "unknown_label": left_out_label,
    }

    if not unknown_only:
        filtered = query_df.copy().reset_index(drop=True)
        if max_query_rows is not None:
            filtered = filtered.iloc[: int(max_query_rows)].copy().reset_index(drop=True)
        stats["query_rows_after"] = int(len(filtered))
        return filtered, stats

    if left_out_label is None:
        raise ValueError("--unknown-only requires --left-out-label to be set.")

    query_attack = one_hot_to_attack_class(query_df, label_cols)
    keep_mask = query_attack == left_out_label
    filtered = query_df.loc[keep_mask].copy().reset_index(drop=True)

    if max_query_rows is not None:
        filtered = filtered.iloc[: int(max_query_rows)].copy().reset_index(drop=True)
    stats["query_rows_after"] = int(len(filtered))
    if len(filtered) == 0:
        raise ValueError(
            f"After applying --unknown-only for label {left_out_label!r}, the query split is empty."
        )
    return filtered, stats


# -----------------------------------------------------------------------------
# Feature partitioning and relationalization
# -----------------------------------------------------------------------------


def infer_cidds_feature_groups(
    df: pd.DataFrame,
    label_cols: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    src_ip_col = "Src IP Addr"
    dst_ip_col = "Dst IP Addr"
    if src_ip_col not in df.columns or dst_ip_col not in df.columns:
        raise ValueError("CIDDS parquet must contain 'Src IP Addr' and 'Dst IP Addr' columns.")

    src_host_cols = [c for c in df.columns if re.fullmatch(r"ipsrc_\d+", str(c))]
    dst_host_cols = [c for c in df.columns if re.fullmatch(r"ipdst_\d+", str(c))]

    excluded = set(label_cols) | {src_ip_col, dst_ip_col} | set(src_host_cols) | set(dst_host_cols)
    flow_feature_cols = [c for c in df.columns if c not in excluded]
    return src_host_cols, dst_host_cols, flow_feature_cols


@dataclass
class FlowBuildResult:
    flows_df: pd.DataFrame
    query_ids: List[str]
    y_true: pd.Series
    labels: List[str]
    column_name_map: Dict[str, str]


@dataclass
class PreparedKumoTables:
    hosts_df: pd.DataFrame
    flows_df: pd.DataFrame
    query_ids: List[str]
    y_true: pd.Series
    labels: List[str]
    feature_rename_map: Dict[str, str]
    build_stats: Dict[str, object]
    noise_stats: Dict[str, object]



def build_hosts_table(
    context_df: pd.DataFrame,
    query_df: pd.DataFrame,
    src_host_cols: Sequence[str],
    dst_host_cols: Sequence[str],
) -> pd.DataFrame:
    both = pd.concat([context_df, query_df], axis=0, ignore_index=True)

    src_cols = ["Src IP Addr", *src_host_cols]
    dst_cols = ["Dst IP Addr", *dst_host_cols]

    src_df = both[src_cols].copy()
    src_df.columns = ["host_id", *[f"ip_bit_{i}" for i in range(1, len(src_host_cols) + 1)]]
    src_df["host_id"] = src_df["host_id"].map(_canonicalize_ip)

    dst_df = both[dst_cols].copy()
    dst_df.columns = ["host_id", *[f"ip_bit_{i}" for i in range(1, len(dst_host_cols) + 1)]]
    dst_df["host_id"] = dst_df["host_id"].map(_canonicalize_ip)

    hosts = pd.concat([src_df, dst_df], axis=0, ignore_index=True)
    hosts = hosts.drop_duplicates(subset=["host_id"], keep="first").reset_index(drop=True)

    for col in hosts.columns:
        if col == "host_id":
            hosts[col] = hosts[col].astype("string")
        else:
            hosts[col] = pd.to_numeric(hosts[col], errors="coerce").astype(np.float32)
    return hosts



def build_flows_table(
    context_df: pd.DataFrame,
    query_df: pd.DataFrame,
    label_cols: Sequence[str],
    flow_feature_cols: Sequence[str],
) -> FlowBuildResult:
    context = context_df.copy().reset_index(drop=True)
    query = query_df.copy().reset_index(drop=True)

    context["attack_class"] = one_hot_to_attack_class(context, label_cols)
    query_true = one_hot_to_attack_class(query, label_cols)

    context["flow_id"] = [f"train_{i}" for i in range(len(context))]
    query["flow_id"] = [f"query_{i}" for i in range(len(query))]
    query["attack_class"] = pd.Series(pd.NA, index=query.index, dtype="string")

    keep_cols = [
        "flow_id",
        "Src IP Addr",
        "Dst IP Addr",
        *flow_feature_cols,
        "attack_class",
    ]
    flows = pd.concat([context[keep_cols], query[keep_cols]], axis=0, ignore_index=True)
    flows = flows.rename(columns={"Src IP Addr": "src_host_id", "Dst IP Addr": "dst_host_id"})

    flows["src_host_id"] = flows["src_host_id"].map(_canonicalize_ip).astype("string")
    flows["dst_host_id"] = flows["dst_host_id"].map(_canonicalize_ip).astype("string")
    flows["flow_id"] = flows["flow_id"].astype("string")
    flows["attack_class"] = flows["attack_class"].astype("string")

    base_cols = ["flow_id", "src_host_id", "dst_host_id", "attack_class"]
    feature_cols_raw = [c for c in flows.columns if c not in base_cols]
    rename_map = _build_unique_name_map(feature_cols_raw)
    flows = flows.rename(columns=rename_map)

    for col in flows.columns:
        if col in {"flow_id", "src_host_id", "dst_host_id", "attack_class"}:
            continue
        flows[col] = pd.to_numeric(flows[col], errors="coerce").astype(np.float32)

    return FlowBuildResult(
        flows_df=flows,
        query_ids=[f"query_{i}" for i in range(len(query))],
        y_true=query_true.reset_index(drop=True),
        labels=list(label_cols),
        column_name_map=rename_map,
    )



def _numeric_feature_columns(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    exclude_set = set(exclude)
    cols: List[str] = []
    for col in df.columns:
        if col in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols



def apply_gaussian_noise_to_tables(
    hosts_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    mean: float,
    std: float,
    seed: int,
    noise_hosts: bool,
    noise_flows: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Mirror loader.GaussianNoise conceptually:
        x <- x + N(mean, std)

    Practical adjustments for Kumo:
    - apply only to numeric feature columns
    - keep IDs/labels untouched
    - store results as float32 to avoid payload blow-up
    """
    hosts = hosts_df.copy()
    flows = flows_df.copy()

    rng = np.random.default_rng(seed)

    host_feature_cols = _numeric_feature_columns(hosts, exclude=["host_id"])
    flow_feature_cols = _numeric_feature_columns(
        flows, exclude=["flow_id", "src_host_id", "dst_host_id", "attack_class"]
    )

    if noise_hosts and host_feature_cols:
        host_values = hosts[host_feature_cols].to_numpy(dtype=np.float32, copy=True)
        host_noise = rng.normal(loc=mean, scale=std, size=host_values.shape).astype(np.float32)
        hosts.loc[:, host_feature_cols] = (host_values + host_noise).astype(np.float32)

    if noise_flows and flow_feature_cols:
        flow_values = flows[flow_feature_cols].to_numpy(dtype=np.float32, copy=True)
        flow_noise = rng.normal(loc=mean, scale=std, size=flow_values.shape).astype(np.float32)
        flows.loc[:, flow_feature_cols] = (flow_values + flow_noise).astype(np.float32)

    hosts = _downcast_numeric_frame(hosts, exclude=["host_id"])
    flows = _downcast_numeric_frame(flows, exclude=["flow_id", "src_host_id", "dst_host_id", "attack_class"])

    noise_stats = {
        "enabled": bool(noise_hosts or noise_flows),
        "mean": float(mean),
        "std": float(std),
        "seed": int(seed),
        "noise_hosts": bool(noise_hosts),
        "noise_flows": bool(noise_flows),
        "n_host_feature_cols_noised": int(len(host_feature_cols) if noise_hosts else 0),
        "n_flow_feature_cols_noised": int(len(flow_feature_cols) if noise_flows else 0),
        "host_feature_cols_preview": host_feature_cols[:10] if noise_hosts else [],
        "flow_feature_cols_preview": flow_feature_cols[:10] if noise_flows else [],
        "dtype_after_noise_hosts": "float32",
        "dtype_after_noise_flows": "float32",
    }
    return hosts, flows, noise_stats



def prepare_kumo_tables_for_split(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    label_cols: Sequence[str],
    left_out_label: Optional[str],
    unknown_only: bool,
    max_query_rows: Optional[int] = None,
    add_noise: bool = False,
    noise_mean: float = 0.0,
    noise_std: float = 0.05,
    noise_seed: int = 42,
    noise_hosts: bool = False,
    noise_flows: bool = True,
) -> PreparedKumoTables:
    context_df, train_stats = filter_train_leave_one_out(
        train_df=train_df,
        label_cols=label_cols,
        left_out_label=left_out_label,
    )
    query_eval_df, query_stats = filter_query_unknown_only(
        query_df=query_df,
        label_cols=label_cols,
        left_out_label=left_out_label,
        unknown_only=unknown_only,
        max_query_rows=max_query_rows,
    )

    src_host_cols, dst_host_cols, flow_feature_cols = infer_cidds_feature_groups(
        train_df, label_cols=label_cols
    )

    hosts_df = build_hosts_table(
        context_df=context_df,
        query_df=query_eval_df,
        src_host_cols=src_host_cols,
        dst_host_cols=dst_host_cols,
    )
    flow_result = build_flows_table(
        context_df=context_df,
        query_df=query_eval_df,
        label_cols=label_cols,
        flow_feature_cols=flow_feature_cols,
    )

    hosts_df = _downcast_numeric_frame(hosts_df, exclude=["host_id"])
    flows_df = _downcast_numeric_frame(
        flow_result.flows_df,
        exclude=["flow_id", "src_host_id", "dst_host_id", "attack_class"],
    )

    if add_noise:
        hosts_df, flows_df, noise_stats = apply_gaussian_noise_to_tables(
            hosts_df=hosts_df,
            flows_df=flows_df,
            mean=noise_mean,
            std=noise_std,
            seed=noise_seed,
            noise_hosts=noise_hosts,
            noise_flows=noise_flows,
        )
    else:
        noise_stats = {
            "enabled": False,
            "mean": float(noise_mean),
            "std": float(noise_std),
            "seed": int(noise_seed),
            "noise_hosts": bool(noise_hosts),
            "noise_flows": bool(noise_flows),
            "n_host_feature_cols_noised": 0,
            "n_flow_feature_cols_noised": 0,
            "host_feature_cols_preview": [],
            "flow_feature_cols_preview": [],
            "dtype_after_noise_hosts": "float32",
            "dtype_after_noise_flows": "float32",
        }

    build_stats = {
        **train_stats,
        **query_stats,
        "n_hosts": int(len(hosts_df)),
        "n_flows_total": int(len(flows_df)),
        "n_query_rows": int(len(flow_result.query_ids)),
        "labels": list(flow_result.labels),
    }

    return PreparedKumoTables(
        hosts_df=hosts_df,
        flows_df=flows_df,
        query_ids=flow_result.query_ids,
        y_true=flow_result.y_true,
        labels=flow_result.labels,
        feature_rename_map=flow_result.column_name_map,
        build_stats=build_stats,
        noise_stats=noise_stats,
    )


# -----------------------------------------------------------------------------
# Kumo setup and inference
# -----------------------------------------------------------------------------


def init_kumo(api_key: Optional[str], url: Optional[str]) -> None:
    if api_key is None:
        api_key = os.environ.get("KUMO_API_KEY")
    if not api_key:
        raise ValueError("Kumo API key missing. Provide --api-key or set KUMO_API_KEY.")

    from kumoai.experimental import rfm  # type: ignore

    os.environ["KUMO_API_KEY"] = api_key
    os.environ["KUMO_URL"] = url or os.environ.get("KUMO_URL", DEFAULT_KUMO_URL)
    rfm.init()



def create_kumo_graph(hosts_df: pd.DataFrame, flows_df: pd.DataFrame, verbose: bool = True):
    from kumoai.experimental import rfm  # type: ignore

    hosts = rfm.LocalTable(df=hosts_df, name="hosts", primary_key="host_id")
    flows = rfm.LocalTable(df=flows_df, name="flows", primary_key="flow_id")

    hosts.infer_metadata(verbose=verbose)
    flows.infer_metadata(verbose=verbose)

    graph = rfm.Graph([hosts, flows])
    graph.link("flows", "src_host_id", "hosts")
    graph.link("flows", "dst_host_id", "hosts")
    graph.validate()
    return graph



def _normalize_kumo_predictions(pred_obj) -> pd.DataFrame:
    pred_df = _to_prediction_df(pred_obj).copy()
    pred_df.columns = [str(c) for c in pred_df.columns]

    long_required = {"ENTITY", "CLASS", "SCORE", "PREDICTED"}
    if long_required.issubset(set(pred_df.columns)):
        df = pred_df.rename(columns={"ENTITY": "flow_id"}).copy()
        df["flow_id"] = df["flow_id"].astype("string")
        df["CLASS"] = df["CLASS"].astype("string")
        df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce").astype(np.float32)
        df["PREDICTED"] = df["PREDICTED"].astype(bool)

        pred_rows = df[df["PREDICTED"] == True][["flow_id", "CLASS"]].drop_duplicates("flow_id")
        pred_rows = pred_rows.rename(columns={"CLASS": "y_pred"})

        if pred_rows["flow_id"].nunique() != df["flow_id"].nunique():
            idx = df.groupby("flow_id")["SCORE"].idxmax()
            fallback = df.loc[idx, ["flow_id", "CLASS"]].rename(columns={"CLASS": "y_pred"})
            pred_rows = fallback.merge(pred_rows, on="flow_id", how="left", suffixes=("_fb", ""))
            pred_rows["y_pred"] = pred_rows["y_pred"].fillna(pred_rows["y_pred_fb"])
            pred_rows = pred_rows[["flow_id", "y_pred"]]

        score_wide = (
            df.pivot_table(index="flow_id", columns="CLASS", values="SCORE", aggfunc="first")
            .reset_index()
        )
        score_wide.columns = [
            "flow_id" if c == "flow_id" else f"score_{c}"
            for c in score_wide.columns
        ]
        for col in score_wide.columns:
            if col != "flow_id":
                score_wide[col] = pd.to_numeric(score_wide[col], errors="coerce").astype(np.float32)
        out = pred_rows.merge(score_wide, on="flow_id", how="left")
        return out

    id_candidates = [
        c for c in pred_df.columns
        if c == "flow_id" or c.endswith(".flow_id") or c in {"ENTITY", "entity"}
    ]
    if id_candidates:
        pred_df = pred_df.rename(columns={id_candidates[0]: "flow_id"})
        pred_df["flow_id"] = pred_df["flow_id"].astype("string")
        return pred_df

    raise ValueError(
        f"Unsupported Kumo prediction schema. shape={pred_df.shape}, columns={list(pred_df.columns)}"
    )


@dataclass
class InferenceResult:
    raw_predictions: pd.DataFrame
    merged: pd.DataFrame
    metrics: Dict[str, object]



def _run_predict_once(model, query: str, query_ids: Sequence[str], batch_size: int, num_retries: int, verbose: bool):
    with model.batch_mode(batch_size=batch_size, num_retries=num_retries):
        return model.predict(query, indices=list(query_ids), verbose=verbose)



def _predict_with_batch_backoff(model, query: str, query_ids: Sequence[str], batch_size: int, num_retries: int, verbose: bool):
    current_batch_size = int(batch_size)
    last_err: Optional[Exception] = None

    while current_batch_size >= 1:
        try:
            if verbose:
                print(f"Running Kumo inference with batch_size={current_batch_size}")
            pred_obj = _run_predict_once(
                model=model,
                query=query,
                query_ids=query_ids,
                batch_size=current_batch_size,
                num_retries=num_retries,
                verbose=verbose,
            )
            return pred_obj, current_batch_size
        except Exception as e:
            last_err = e
            msg = str(e)
            if KUMO_CONTEXT_LIMIT_SNIPPET not in msg or current_batch_size == 1:
                raise
            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"Kumo context exceeded at batch_size={current_batch_size}. Retrying with batch_size={next_batch_size}."
            )
            current_batch_size = next_batch_size

    assert last_err is not None
    raise last_err



def run_kumo_inference(
    hosts_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    query_ids: Sequence[str],
    y_true: Sequence[str],
    labels: Sequence[str],
    batch_size: int,
    num_retries: int,
    verbose: bool,
    auto_batch_backoff: bool,
    debug_dir: Optional[Path] = None,
) -> InferenceResult:
    from kumoai.experimental import rfm  # type: ignore

    graph = create_kumo_graph(hosts_df, flows_df, verbose=verbose)
    model = rfm.KumoRFM(graph, verbose=verbose)

    first_id = query_ids[0]
    query = f"PREDICT flows.attack_class FOR flows.flow_id={_pql_literal(first_id)}"

    if auto_batch_backoff:
        pred_obj, final_batch_size = _predict_with_batch_backoff(
            model=model,
            query=query,
            query_ids=query_ids,
            batch_size=batch_size,
            num_retries=num_retries,
            verbose=verbose,
        )
    else:
        pred_obj = _run_predict_once(
            model=model,
            query=query,
            query_ids=query_ids,
            batch_size=batch_size,
            num_retries=num_retries,
            verbose=verbose,
        )
        final_batch_size = int(batch_size)

    raw_pred_df = _to_prediction_df(pred_obj)
    if debug_dir is not None:
        _ensure_dir(debug_dir)
        raw_pred_df.to_csv(debug_dir / "raw_predictions_debug.csv", index=True)
        with open(debug_dir / "raw_predictions_debug.txt", "w", encoding="utf-8") as f:
            f.write(f"type={type(pred_obj)}\n")
            f.write(f"shape={raw_pred_df.shape}\n")
            f.write(f"columns={list(raw_pred_df.columns)}\n")
            f.write(f"index_name={raw_pred_df.index.name}\n")
            f.write(f"index_type={type(raw_pred_df.index)}\n")
            f.write(f"final_batch_size={final_batch_size}\n")
            f.write("head=\n")
            f.write(raw_pred_df.head(30).to_string())

    pred_df = _normalize_kumo_predictions(pred_obj)

    alignment = pd.DataFrame(
        {
            "flow_id": pd.Series(list(query_ids), dtype="string"),
            "y_true": pd.Series(list(y_true), dtype="string"),
        }
    )
    merged = alignment.merge(pred_df, on="flow_id", how="left")

    if "y_pred" not in merged.columns:
        raise ValueError(f"Normalized predictions do not contain y_pred. Columns={list(merged.columns)}")

    missing = int(merged["y_pred"].isna().sum())
    if missing > 0:
        raise ValueError(f"Missing predictions for {missing}/{len(merged)} query flows after alignment.")

    metrics = {
        "n_query_rows": int(len(merged)),
        "accuracy": float(accuracy_score(merged["y_true"], merged["y_pred"])),
        "macro_f1": float(f1_score(merged["y_true"], merged["y_pred"], average="macro")),
        "weighted_f1": float(f1_score(merged["y_true"], merged["y_pred"], average="weighted")),
        "labels": list(labels),
        "classification_report": classification_report(
            merged["y_true"],
            merged["y_pred"],
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            merged["y_true"],
            merged["y_pred"],
            labels=list(labels),
        ).tolist(),
        "final_batch_size": int(final_batch_size),
    }
    return InferenceResult(raw_predictions=pred_df, merged=merged, metrics=metrics)


# -----------------------------------------------------------------------------
# Saving and CLI
# -----------------------------------------------------------------------------


def save_prepared_tables(prepared: PreparedKumoTables, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    prepared.hosts_df.to_parquet(out_dir / "hosts.parquet", index=False)
    prepared.flows_df.to_parquet(out_dir / "flows.parquet", index=False)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": prepared.labels,
                "n_hosts": int(len(prepared.hosts_df)),
                "n_flows_total": int(len(prepared.flows_df)),
                "n_query_rows": int(len(prepared.query_ids)),
                "query_ids_preview": prepared.query_ids[:10],
                "flow_feature_rename_map": prepared.feature_rename_map,
                "build_stats": prepared.build_stats,
                "noise_stats": prepared.noise_stats,
            },
            f,
            indent=2,
        )



def save_inference_outputs(result: InferenceResult, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    result.raw_predictions.to_parquet(out_dir / "predictions_raw.parquet", index=False)
    result.merged.to_parquet(out_dir / "predictions_merged.parquet", index=False)
    result.merged.to_csv(out_dir / "predictions_merged.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KumoRFM leave-one-out evaluation on CIDDS with optional Gaussian noise.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory holding df_train/df_val/df_test.parquet")
    parser.add_argument("--query-split", type=str, default="test", choices=["val", "test"], help="Held-out split to predict")
    parser.add_argument("--left-out-label", type=str, default=None, choices=KNOWN_CIDDS_LABELS, help="Remove this class from df_train before building Kumo context")
    parser.add_argument("--unknown-only", action="store_true", help="Predict/evaluate only rows of the left-out class in the query split")
    parser.add_argument("--add-noise", action="store_true", help="Add Gaussian noise before Kumo graph creation")
    parser.add_argument("--noise-mean", type=float, default=0.0, help="Gaussian noise mean; matches loader.GaussianNoise(mean=...)")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Gaussian noise std; matches loader.GaussianNoise(std=...).")
    parser.add_argument("--noise-seed", type=int, default=42, help="Random seed for reproducible noise generation")
    parser.add_argument("--noise-hosts", action="store_true", help="Also noise numeric host features. Default is flow features only.")
    parser.add_argument("--no-noise-flows", action="store_true", help="Disable noise on flow features.")
    parser.add_argument("--out-dir", type=str, default="./artifacts/kumo_cidds_loo_noise", help="Where to save exported tables and predictions")
    parser.add_argument("--prepare-only", action="store_true", help="Only build and export Kumo-ready tables; do not call Kumo")
    parser.add_argument("--max-query-rows", type=int, default=None, help="Optional cap on number of query rows after filtering")
    parser.add_argument("--batch-size", type=int, default=1000, help="Initial batch size for KumoRFM.predict()")
    parser.add_argument("--num-retries", type=int, default=1, help="Retries for failed Kumo requests")
    parser.add_argument("--disable-auto-batch-backoff", action="store_true", help="Disable automatic retry with smaller batch sizes on Kumo 30MB context errors")
    parser.add_argument("--api-key", type=str, default=None, help="Kumo API key (fallback: KUMO_API_KEY env var)")
    parser.add_argument("--kumo-url", type=str, default=None, help=f"Kumo API URL (default: {DEFAULT_KUMO_URL})")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    if args.unknown_only and args.left_out_label is None:
        raise ValueError("--unknown-only requires --left-out-label.")
    if args.noise_std < 0:
        raise ValueError("--noise-std must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    noise_flows = not args.no_noise_flows
    if args.add_noise and not (args.noise_hosts or noise_flows):
        raise ValueError("--add-noise was set, but both host and flow noise are disabled.")

    data_dir = Path(args.data_dir)
    tag = f"query_{args.query_split}"
    if args.left_out_label is not None:
        tag += f"__loo_{args.left_out_label}"
    if args.unknown_only:
        tag += "__unknown_only"
    if args.add_noise:
        tag += f"__noise_m{args.noise_mean:g}_s{args.noise_std:g}_seed{args.noise_seed}"
        if args.noise_hosts:
            tag += "__hosts"
        if noise_flows:
            tag += "__flows"
    out_dir = Path(args.out_dir) / tag
    _ensure_dir(out_dir)

    splits = load_cidds_splits(data_dir)
    query_df = splits.val if args.query_split == "val" else splits.test

    prepared = prepare_kumo_tables_for_split(
        train_df=splits.train,
        query_df=query_df,
        label_cols=splits.label_cols,
        left_out_label=args.left_out_label,
        unknown_only=args.unknown_only,
        max_query_rows=args.max_query_rows,
        add_noise=args.add_noise,
        noise_mean=args.noise_mean,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        noise_hosts=args.noise_hosts,
        noise_flows=noise_flows,
    )

    save_prepared_tables(prepared, out_dir)

    print(f"Saved Kumo-ready tables to: {out_dir}")
    print(f"  hosts: {len(prepared.hosts_df):,}")
    print(f"  flows: {len(prepared.flows_df):,} (context labeled + {len(prepared.query_ids):,} query unlabeled)")
    print(f"  labels: {prepared.labels}")
    if args.left_out_label is not None:
        print(
            f"  leave-one-out: removed {prepared.build_stats['n_left_out_removed']:,} train rows of class {args.left_out_label!r}"
        )
    if args.unknown_only:
        print(f"  query filter: evaluating only unknown class {args.left_out_label!r}")
    if args.add_noise:
        print(
            "  gaussian noise: "
            f"mean={prepared.noise_stats['mean']}, std={prepared.noise_stats['std']}, seed={prepared.noise_stats['seed']}"
        )
        print(
            "  noised feature columns: "
            f"hosts={prepared.noise_stats['n_host_feature_cols_noised']}, "
            f"flows={prepared.noise_stats['n_flow_feature_cols_noised']}"
        )
        print(
            "  noise targets: "
            f"hosts={prepared.noise_stats['noise_hosts']}, flows={prepared.noise_stats['noise_flows']}"
        )

    if args.prepare_only:
        print("Preparation complete. Skipping Kumo inference because --prepare-only was set.")
        return

    init_kumo(api_key=args.api_key, url=args.kumo_url)
    result = run_kumo_inference(
        hosts_df=prepared.hosts_df,
        flows_df=prepared.flows_df,
        query_ids=prepared.query_ids,
        y_true=prepared.y_true.tolist(),
        labels=prepared.labels,
        batch_size=args.batch_size,
        num_retries=args.num_retries,
        verbose=verbose,
        auto_batch_backoff=not args.disable_auto_batch_backoff,
        debug_dir=out_dir / "debug",
    )
    save_inference_outputs(result, out_dir)

    print("\nEvaluation on query split")
    print(f"  accuracy         : {result.metrics['accuracy']:.4f}")
    print(f"  macro_f1         : {result.metrics['macro_f1']:.4f}")
    print(f"  weighted_f1      : {result.metrics['weighted_f1']:.4f}")
    print(f"  final_batch_size : {result.metrics['final_batch_size']}")
    print(f"Saved predictions and metrics to: {out_dir}")


if __name__ == "__main__":
    main()
