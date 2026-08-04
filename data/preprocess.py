import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, LabelEncoder

import scipy.sparse as sp
import numpy as np
import torch

import json
import os

from utils.unsw_utils import _pick_first_existing, _safe_uint8, _ipv4_features, _to_num, decode_nf_tcp_flags_0_31, _ipv4_features, sanitize_for_scaling


def preprocess_CIDDS() -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sketches insightful figures for CIDDS. Saves train, val, and test splits as parquet files. Adopted from
    https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python/blob/main/Chapter16/chapter16
    .ipynb
    :return: df_train, df_val, df_test
    """

    df = pd.read_csv("./data/CIDDS/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv")
    print("RAW DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    # Drop unnecessary colums
    df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos', 'class', 'attackID', 'attackDescription'])
    df['attackType'] = df['attackType'].replace('---', 'benign')
    df['Date first seen'] = pd.to_datetime(df['Date first seen'])
    print("NEW DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    # Count labels
    count_labels = df['attackType'].value_counts() / len(df) * 100
    print(count_labels)
    plt.pie(count_labels[:3], labels=df['attackType'].unique()[:3], autopct='%.0f%%')
    plt.savefig("count_label", dpi=300, bbox_inches='tight')

    # Feature Distribution
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
    df['Duration'].hist(ax=ax1)
    ax1.set_xlabel("Duration")
    df['Packets'].hist(ax=ax2)
    ax2.set_xlabel("Number of packets")
    pd.to_numeric(df['Bytes'], errors='coerce').hist(ax=ax3)
    ax3.set_xlabel("Number of bytes")
    plt.savefig("feature_distribution", dpi=300, bbox_inches='tight')

    # Preprocessing the CIDDS-001 dataset

    df['weekday'] = df['Date first seen'].dt.weekday
    df = pd.get_dummies(df, columns=['weekday']).rename(columns={'weekday_0': 'Monday',
                                                                 'weekday_1': 'Tuesday',
                                                                 'weekday_2': 'Wednesday',
                                                                 'weekday_3': 'Thursday',
                                                                 'weekday_4': 'Friday',
                                                                 'weekday_5': 'Saturday',
                                                                 'weekday_6': 'Sunday',
                                                                 })

    df['daytime'] = (df['Date first seen'].dt.second + df['Date first seen'].dt.minute * 60 + df[
        'Date first seen'].dt.hour * 60 * 60) / (24 * 60 * 60)

    def one_hot_flags(input):
        return [1 if char1 == char2 else 0 for char1, char2 in zip('APRSF', input[1:])]

    df = df.reset_index(drop=True)
    ohe_flags = one_hot_flags(df['Flags'].to_numpy())
    ohe_flags = df['Flags'].apply(one_hot_flags).to_list()
    df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(ohe_flags, columns=['ACK', 'PSH', 'RST', 'SYN', 'FIN'])
    df = df.drop(columns=['Date first seen', 'Flags'])
    print("PREPROCESSED DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    temp = pd.DataFrame()
    temp['SrcIP'] = df['Src IP Addr'].astype(str)
    temp['SrcIP'][~temp['SrcIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['SrcIP'].str.split('.', expand=True).rename(columns={2: 'ipsrc3', 3: 'ipsrc4'}).astype(int)[
        ['ipsrc3', 'ipsrc4']]
    temp['ipsrc'] = temp['ipsrc3'].apply(lambda x: format(x, "b").zfill(8)) + temp['ipsrc4'].apply(
        lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipsrc'].str.split('', expand=True)
                 .drop(columns=[0, 17])
                 .rename(columns=dict(enumerate([f'ipsrc_{i}' for i in range(17)])))
                 .astype('int32'))
    print(df.head(5))

    temp = pd.DataFrame()
    temp['DstIP'] = df['Dst IP Addr'].astype(str)
    temp['DstIP'][~temp['DstIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['DstIP'].str.split('.', expand=True).rename(columns={2: 'ipdst3', 3: 'ipdst4'}).astype(int)[
        ['ipdst3', 'ipdst4']]
    temp['ipdst'] = temp['ipdst3'].apply(lambda x: format(x, "b").zfill(8)) \
                    + temp['ipdst4'].apply(lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipdst'].str.split('', expand=True)
                 .drop(columns=[0, 17])
                 .rename(columns=dict(enumerate([f'ipdst_{i}' for i in range(17)])))
                 .astype('int32'))
    print(df.head(5))

    m_index = df[pd.to_numeric(df['Bytes'], errors='coerce').isnull() == True].index
    df['Bytes'].loc[m_index] = df['Bytes'].loc[m_index].apply(lambda x: 10e6 * float(x.strip().split()[0]))
    df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce', downcast='integer')

    df = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Proto', 'attackType'])
    print(df.head(5))

    labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']
    df_train, df_test = train_test_split(df, random_state=0, test_size=0.2, stratify=df[labels])
    df_val, df_test = train_test_split(df_test, random_state=0, test_size=0.5, stratify=df_test[labels])

    scaler = PowerTransformer()
    df_train[['Duration', 'Packets', 'Bytes']] = scaler.fit_transform(df_train[['Duration', 'Packets', 'Bytes']])
    df_val[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_val[['Duration', 'Packets', 'Bytes']])
    df_test[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_test[['Duration', 'Packets', 'Bytes']])

    df_train.to_parquet('./data/CIDDS/df_train.parquet')
    df_val.to_parquet('./data/CIDDS/df_val.parquet')
    df_test.to_parquet('./data/CIDDS/df_test.parquet')

    print(df_train[df_train['benign'] == 1].head())

    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
    df_train['Duration'].hist(ax=ax1)
    ax1.set_xlabel("Duration")
    df_train['Packets'].hist(ax=ax2)
    ax2.set_xlabel("Number of packets")
    df_train['Bytes'].hist(ax=ax3)
    ax3.set_xlabel("Number of bytes")
    plt.savefig("scaled_feature_distribution", dpi=300, bbox_inches='tight')

    return df_train, df_val, df_test


def preprocess_nf_unsw_nb15_v3_compact(
    raw_path: str,
    out_dir: str = "./data/NF-UNSW-NB15-v3-compact",
    random_state: int = 0,
    test_size: float = 0.2,
    val_size_within_test: float = 0.5,
    hash_bins: int = 2**16,
    top_dst_ports: int = 20,
    top_src_ports: int = 10,
    top_l7_protos: int = 50,
):
    """
    Build a compact feature table for NF-UNSW-NB15-v3.

    Key fix vs. "terrible MLP performance":
      - DO NOT feed raw categorical IDs (PROTOCOL, L4_*_PORT, L7_PROTO) as continuous numbers.
      - Instead: TRAIN-driven one-hot / bucket encodings, then drop the raw ID columns.

    Saves:
      df_train.parquet, df_val.parquet, df_test.parquet
      attack_labels.json  (list of label column names)
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(raw_path)
    print("RAW SHAPE:", df.shape)
    print(df.head())

    # ---- Required label
    if "Attack" not in df.columns:
        raise ValueError("This pipeline requires an 'Attack' column for the flow class label.")
    df["Attack"] = df["Attack"].astype(str).fillna("unknown")
    df.loc[df["Attack"].isin(["nan", "None", ""]), "Attack"] = "unknown"

    # normalize benign naming (force lowercase benign)
    df["Attack"] = df["Attack"].str.strip()
    df.loc[df["Attack"].str.lower().isin(["benign", "normal", "background"]), "Attack"] = "benign"

    # ---- Timestamp -> weekday/daytime
    start_col = _pick_first_existing(df, ["FLOW_START_MILLISECONDS", "flow_start_milliseconds", "FLOW_START_MS"])
    end_col = _pick_first_existing(df, ["FLOW_END_MILLISECONDS", "flow_end_milliseconds", "FLOW_END_MS"])

    if start_col:
        dt = pd.to_datetime(df[start_col], unit="ms", errors="coerce")
    elif end_col:
        dt = pd.to_datetime(df[end_col], unit="ms", errors="coerce")
    else:
        dt = pd.to_datetime(np.arange(len(df)), unit="s", origin="unix", errors="coerce")

    dt = dt.fillna(pd.Timestamp("1970-01-01"))

    # Build weekday dummies, but keep ONLY the days that actually appear in the dataset.
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_idx = dt.dt.weekday  # 0..6
    weekday_ohe = pd.get_dummies(weekday_idx)
    weekday_ohe.columns = [day_names[i] for i in weekday_ohe.columns]
    weekday_ohe = weekday_ohe.astype(bool)
    weekday_ohe = weekday_ohe.loc[:, weekday_ohe.any(axis=0)]  # drop days absent

    daytime = ((dt.dt.second + dt.dt.minute * 60 + dt.dt.hour * 3600) / (24 * 3600)).astype("float32")

    # ---- Duration
    if "FLOW_DURATION_MILLISECONDS" in df.columns:
        duration = (pd.to_numeric(df["FLOW_DURATION_MILLISECONDS"], errors="coerce").fillna(0) / 1000.0).astype("float32")
    else:
        duration = pd.Series(np.zeros(len(df), dtype="float32"))

    # ---- Packets/Bytes totals
    in_pkts = pd.to_numeric(df["IN_PKTS"], errors="coerce").fillna(0).astype("float32") if "IN_PKTS" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"))
    out_pkts = pd.to_numeric(df["OUT_PKTS"], errors="coerce").fillna(0).astype("float32") if "OUT_PKTS" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"))
    packets = (in_pkts + out_pkts).astype("float32")

    in_bytes = pd.to_numeric(df["IN_BYTES"], errors="coerce").fillna(0).astype("float32") if "IN_BYTES" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"))
    out_bytes = pd.to_numeric(df["OUT_BYTES"], errors="coerce").fillna(0).astype("float32") if "OUT_BYTES" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"))
    bytes_total = (in_bytes + out_bytes).astype("float32")

    # ---- IP strings (must-have)
    src_ip_col = _pick_first_existing(df, ["IPV4_SRC_ADDR", "IPv4_SRC_ADDR", "src_ip", "SRC_IP"])
    dst_ip_col = _pick_first_existing(df, ["IPV4_DST_ADDR", "IPv4_DST_ADDR", "dst_ip", "DST_IP"])
    src_ip = df[src_ip_col] if src_ip_col else pd.Series(["0.0.0.0"] * len(df))
    dst_ip = df[dst_ip_col] if dst_ip_col else pd.Series(["0.0.0.0"] * len(df))

    # ---- Build base output (must-have)
    out = pd.DataFrame(
        {
            "Duration": duration,
            "Src IP Addr": src_ip.astype(str),
            "Dst IP Addr": dst_ip.astype(str),
            "Packets": packets,
            "Bytes": bytes_total,
            "daytime": daytime,
        }
    ).reset_index(drop=True)

    out = pd.concat([out, weekday_ohe.reset_index(drop=True)], axis=1)

    # ---- Compact IP numeric features (full 4 octets + int32 + hash)
    out = pd.concat(
        [
            out,
            _ipv4_features(out["Src IP Addr"], "srcip", hash_bins=hash_bins).reset_index(drop=True),
            _ipv4_features(out["Dst IP Addr"], "dstip", hash_bins=hash_bins).reset_index(drop=True),
        ],
        axis=1,
    )

    # ---- Additional compact features (raw IDs kept only as intermediate columns)
    # We will encode these properly *after* train/val/test split so that we can
    # pick top-K categories based on TRAIN ONLY.
    if "PROTOCOL" in df.columns:
        out["PROTOCOL"] = pd.to_numeric(df["PROTOCOL"], errors="coerce").fillna(0).clip(0, 255).astype("int32")
    else:
        out["PROTOCOL"] = 0

    if "L4_SRC_PORT" in df.columns:
        out["L4_SRC_PORT"] = pd.to_numeric(df["L4_SRC_PORT"], errors="coerce").fillna(0).clip(0, 65535).astype("int32")
    else:
        out["L4_SRC_PORT"] = 0

    if "L4_DST_PORT" in df.columns:
        out["L4_DST_PORT"] = pd.to_numeric(df["L4_DST_PORT"], errors="coerce").fillna(0).clip(0, 65535).astype("int32")
    else:
        out["L4_DST_PORT"] = 0

    if "L7_PROTO" in df.columns:
        out["L7_PROTO"] = pd.to_numeric(df["L7_PROTO"], errors="coerce").fillna(0).clip(0, 65535).astype("int32")
    else:
        out["L7_PROTO"] = 0

    out["IN_PKTS"] = in_pkts
    out["OUT_PKTS"] = out_pkts
    out["IN_BYTES"] = in_bytes
    out["OUT_BYTES"] = out_bytes

    eps = 1e-6
    out["BYTE_IN_RATIO"] = (out["IN_BYTES"] / (out["IN_BYTES"] + out["OUT_BYTES"] + eps)).astype("float32")
    out["PKT_IN_RATIO"] = (out["IN_PKTS"] / (out["IN_PKTS"] + out["OUT_PKTS"] + eps)).astype("float32")

    for col in ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
                "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
        else:
            out[col] = 0.0

    flags_feat = decode_nf_tcp_flags_0_31(df, flags_col="TCP_FLAGS", proto_col="PROTOCOL").reset_index(drop=True)
    out = pd.concat([out, flags_feat], axis=1)

    for col in ["MIN_TTL", "MAX_TTL", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
        else:
            out[col] = 0.0

    # ---- One-hot label columns from Attack (boolean) with benign as index 0
    attack_series = df["Attack"].astype(str)
    unique_attacks = sorted(attack_series.unique())
    if "benign" in unique_attacks:
        unique_attacks = ["benign"] + [a for a in unique_attacks if a != "benign"]
    else:
        raise ValueError("No 'benign' label found after normalization. Check df['Attack'] values.")

    attack_cat = pd.Categorical(attack_series, categories=unique_attacks, ordered=True)
    attack_ohe = pd.get_dummies(attack_cat).astype(bool)

    label_cols = list(attack_ohe.columns)
    assert label_cols[0].lower() == "benign", f"Expected benign first, got {label_cols[0]}"

    out = pd.concat([out.reset_index(drop=True), attack_ohe.reset_index(drop=True)], axis=1)

    with open(os.path.join(out_dir, "attack_labels.json"), "w") as f:
        json.dump(label_cols, f, indent=2)

    # ---- Splits (stratify by Attack)
    y_strat = df["Attack"].astype(str)
    df_train, df_test = train_test_split(out, test_size=test_size, random_state=random_state, stratify=y_strat)
    y_strat_test = y_strat.loc[df_test.index]
    df_val, df_test = train_test_split(
        df_test, test_size=val_size_within_test, random_state=random_state, stratify=y_strat_test
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # ---- Encode categorical-ish raw ID columns properly (TRAIN-driven)
    def _add_proto_ohe(s: pd.Series) -> pd.DataFrame:
        p = pd.to_numeric(s, errors="coerce").fillna(0).astype("int32")
        is_tcp = (p == 6).astype("int8")
        is_udp = (p == 17).astype("int8")
        is_icmp = ((p == 1) | (p == 58)).astype("int8")
        is_other = ((is_tcp == 0) & (is_udp == 0) & (is_icmp == 0)).astype("int8")
        return pd.DataFrame({
            "IS_UDP": is_udp,
            "IS_ICMP": is_icmp,
            "IS_OTHER": is_other,
        })

    def _add_port_buckets(df_split: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
        x = pd.to_numeric(df_split[col], errors="coerce").fillna(0).clip(0, 65535).astype("int32")
        return pd.DataFrame({
            f"{prefix}_PORT_ZERO": (x == 0).astype("int8"),
            f"{prefix}_PORT_WELLKNOWN": ((x >= 1) & (x <= 1023)).astype("int8"),
            f"{prefix}_PORT_REGISTERED": ((x >= 1024) & (x <= 49151)).astype("int8"),
            f"{prefix}_PORT_EPHEMERAL": ((x >= 49152) & (x <= 65535)).astype("int8"),
        })

    # Pick top-K categories from TRAIN ONLY
    top_dst = (
        df_train["L4_DST_PORT"].value_counts().head(int(top_dst_ports)).index.to_list()
        if "L4_DST_PORT" in df_train.columns else []
    )
    top_src = (
        df_train["L4_SRC_PORT"].value_counts().head(int(top_src_ports)).index.to_list()
        if "L4_SRC_PORT" in df_train.columns else []
    )
    top_l7 = (
        df_train["L7_PROTO"].value_counts().head(int(top_l7_protos)).index.to_list()
        if "L7_PROTO" in df_train.columns else []
    )

    def _apply_categorical_encodings(df_split: pd.DataFrame) -> pd.DataFrame:
        df_split = df_split.copy()

        # Protocol indicators (keep IS_TCP from tcp-flags decoder, add others)
        proto_ohe = _add_proto_ohe(df_split.get("PROTOCOL", 0))
        df_split = pd.concat([df_split, proto_ohe], axis=1)

        # Port buckets
        if "L4_SRC_PORT" in df_split.columns:
            df_split = pd.concat([df_split, _add_port_buckets(df_split, "L4_SRC_PORT", "SRC")], axis=1)
        if "L4_DST_PORT" in df_split.columns:
            df_split = pd.concat([df_split, _add_port_buckets(df_split, "L4_DST_PORT", "DST")], axis=1)

        # Top-N port one-hots
        for p in top_dst:
            df_split[f"DST_PORT_{int(p)}"] = (df_split["L4_DST_PORT"] == int(p)).astype("int8")
        for p in top_src:
            df_split[f"SRC_PORT_{int(p)}"] = (df_split["L4_SRC_PORT"] == int(p)).astype("int8")

        # L7 proto top-K one-hots + OTHER
        if "L7_PROTO" in df_split.columns:
            for v in top_l7:
                df_split[f"L7_{int(v)}"] = (df_split["L7_PROTO"] == int(v)).astype("int8")
            if len(top_l7) > 0:
                df_split["L7_OTHER"] = (~df_split["L7_PROTO"].isin([int(v) for v in top_l7])).astype("int8")
            else:
                df_split["L7_OTHER"] = 1

        # Drop raw-ID columns (not safe as continuous features)
        for raw in ["PROTOCOL", "L4_SRC_PORT", "L4_DST_PORT", "L7_PROTO"]:
            if raw in df_split.columns:
                df_split = df_split.drop(columns=[raw])
        return df_split

    df_train = _apply_categorical_encodings(df_train)
    df_val = _apply_categorical_encodings(df_val)
    df_test = _apply_categorical_encodings(df_test)

    # ---- Scaling (only continuous columns)
    scale_cols = [
        "Duration", "Packets", "Bytes",
        "IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES",
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "MIN_TTL", "MAX_TTL",
        "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    ]
    scale_cols = [c for c in scale_cols if c in df_train.columns]

    X_train = sanitize_for_scaling(df_train, scale_cols)
    X_val = sanitize_for_scaling(df_val, scale_cols)
    X_test = sanitize_for_scaling(df_test, scale_cols)

    scaler = PowerTransformer()
    df_train.loc[:, scale_cols] = scaler.fit_transform(X_train)
    df_val.loc[:, scale_cols] = scaler.transform(X_val)
    df_test.loc[:, scale_cols] = scaler.transform(X_test)

    # ---- Save parquet splits
    df_train.to_parquet(os.path.join(out_dir, "df_train.parquet"))
    df_val.to_parquet(os.path.join(out_dir, "df_val.parquet"))
    df_test.to_parquet(os.path.join(out_dir, "df_test.parquet"))

    print("✅ Done.")
    print("Weekday columns kept:", list(weekday_ohe.columns))
    print("Shapes:", df_train.shape, df_val.shape, df_test.shape)
    print("Num classes:", len(label_cols))
    print("Saved to:", out_dir)

    return df_train, df_val, df_test

def wo_port_preprocess_nf_unsw_nb15_v3_compact(
    raw_path: str,
    out_dir: str = "./data/NF-UNSW-NB15-v3-compact",
    random_state: int = 0,
    test_size: float = 0.2,
    val_size_within_test: float = 0.5,
    hash_bins: int = 2**16,
    subsample: bool = False,
    subsample_ratio: float = 0.2,
):
    """
    Build a compact feature table for NF-UNSW-NB15-v3.

    This version explicitly DOES NOT USE:
      - L4_SRC_PORT, L4_DST_PORT (ports)
      - L7_PROTO (L7 information)

    It also does NOT create any top-K one-hot columns from them.

    Optional:
      - If subsample=True, the raw dataframe is stratified-subsampled
        according to the class ratios in the 'Attack' column before
        feature construction and train/val/test splitting.

    Important fix:
      - The train/val/test split now carries y_strat through train_test_split
        directly. This avoids index mismatch errors when the original parquet
        has a non-contiguous or non-RangeIndex.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(raw_path)

    # Always reset index immediately after loading.
    # This avoids index mismatch between df, out, and y_strat.
    df = df.reset_index(drop=True)

    # ---- Required label
    if "Attack" not in df.columns:
        raise ValueError("This pipeline requires an 'Attack' column for the flow class label.")

    df["Attack"] = df["Attack"].astype(str).fillna("unknown")
    df.loc[df["Attack"].isin(["nan", "None", ""]), "Attack"] = "unknown"

    # normalize benign naming
    df["Attack"] = df["Attack"].str.strip()
    df.loc[df["Attack"].str.lower().isin(["benign", "normal", "background"]), "Attack"] = "benign"

    # ---- Optional stratified subsampling by Attack
    if subsample:
        if not (0 < subsample_ratio <= 1):
            raise ValueError("subsample_ratio must be in the interval (0, 1].")

        sampled_parts = []
        for attack_name, group in df.groupby("Attack", sort=False):
            n_sample = max(1, int(round(len(group) * subsample_ratio)))
            sampled_parts.append(
                group.sample(
                    n=n_sample,
                    random_state=random_state,
                )
            )

        df = (
            pd.concat(sampled_parts, axis=0)
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )

        print(f"Subsampled dataset with ratio={subsample_ratio}")
        print("Subsampled shape:", df.shape)
        print("Subsampled class ratios:")
        print(df["Attack"].value_counts(normalize=True))

    # ---- Timestamp -> weekday/daytime
    start_col = _pick_first_existing(
        df,
        ["FLOW_START_MILLISECONDS", "flow_start_milliseconds", "FLOW_START_MS"],
    )
    end_col = _pick_first_existing(
        df,
        ["FLOW_END_MILLISECONDS", "flow_end_milliseconds", "FLOW_END_MS"],
    )

    if start_col:
        dt = pd.to_datetime(df[start_col], unit="ms", errors="coerce")
    elif end_col:
        dt = pd.to_datetime(df[end_col], unit="ms", errors="coerce")
    else:
        dt = pd.to_datetime(np.arange(len(df)), unit="s", origin="unix", errors="coerce")

    dt = dt.fillna(pd.Timestamp("1970-01-01"))

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_idx = dt.dt.weekday

    weekday_ohe = pd.get_dummies(weekday_idx)
    weekday_ohe.columns = [day_names[i] for i in weekday_ohe.columns]
    weekday_ohe = weekday_ohe.astype(bool)
    weekday_ohe = weekday_ohe.loc[:, weekday_ohe.any(axis=0)]

    daytime = (
        (dt.dt.second + dt.dt.minute * 60 + dt.dt.hour * 3600) / (24 * 3600)
    ).astype("float32")

    # ---- Duration
    if "FLOW_DURATION_MILLISECONDS" in df.columns:
        duration = (
            pd.to_numeric(df["FLOW_DURATION_MILLISECONDS"], errors="coerce").fillna(0)
            / 1000.0
        ).astype("float32")
    else:
        duration = pd.Series(np.zeros(len(df), dtype="float32"))

    # ---- Packets/Bytes totals
    if "IN_PKTS" in df.columns:
        in_pkts = pd.to_numeric(df["IN_PKTS"], errors="coerce").fillna(0).astype("float32")
    else:
        in_pkts = pd.Series(np.zeros(len(df), dtype="float32"))

    if "OUT_PKTS" in df.columns:
        out_pkts = pd.to_numeric(df["OUT_PKTS"], errors="coerce").fillna(0).astype("float32")
    else:
        out_pkts = pd.Series(np.zeros(len(df), dtype="float32"))

    packets = (in_pkts + out_pkts).astype("float32")

    if "IN_BYTES" in df.columns:
        in_bytes = pd.to_numeric(df["IN_BYTES"], errors="coerce").fillna(0).astype("float32")
    else:
        in_bytes = pd.Series(np.zeros(len(df), dtype="float32"))

    if "OUT_BYTES" in df.columns:
        out_bytes = pd.to_numeric(df["OUT_BYTES"], errors="coerce").fillna(0).astype("float32")
    else:
        out_bytes = pd.Series(np.zeros(len(df), dtype="float32"))

    bytes_total = (in_bytes + out_bytes).astype("float32")

    # ---- IP strings
    src_ip_col = _pick_first_existing(
        df,
        ["IPV4_SRC_ADDR", "IPv4_SRC_ADDR", "src_ip", "SRC_IP"],
    )
    dst_ip_col = _pick_first_existing(
        df,
        ["IPV4_DST_ADDR", "IPv4_DST_ADDR", "dst_ip", "DST_IP"],
    )

    src_ip = df[src_ip_col] if src_ip_col else pd.Series(["0.0.0.0"] * len(df))
    dst_ip = df[dst_ip_col] if dst_ip_col else pd.Series(["0.0.0.0"] * len(df))

    # ---- Build base output
    out = pd.DataFrame(
        {
            "Duration": duration,
            "Src IP Addr": src_ip.astype(str),
            "Dst IP Addr": dst_ip.astype(str),
            "Packets": packets,
            "Bytes": bytes_total,
            "daytime": daytime,
        }
    ).reset_index(drop=True)

    out = pd.concat([out, weekday_ohe.reset_index(drop=True)], axis=1)

    # ---- Compact IP numeric features
    out = pd.concat(
        [
            out,
            _ipv4_features(out["Src IP Addr"], "srcip", hash_bins=hash_bins).reset_index(drop=True),
            _ipv4_features(out["Dst IP Addr"], "dstip", hash_bins=hash_bins).reset_index(drop=True),
        ],
        axis=1,
    )

    # ---- Extra numeric features: NO ports, NO L7
    out["IN_PKTS"] = in_pkts.reset_index(drop=True)
    out["OUT_PKTS"] = out_pkts.reset_index(drop=True)
    out["IN_BYTES"] = in_bytes.reset_index(drop=True)
    out["OUT_BYTES"] = out_bytes.reset_index(drop=True)

    eps = 1e-6

    out["BYTE_IN_RATIO"] = (
        out["IN_BYTES"] / (out["IN_BYTES"] + out["OUT_BYTES"] + eps)
    ).astype("float32")

    out["PKT_IN_RATIO"] = (
        out["IN_PKTS"] / (out["IN_PKTS"] + out["OUT_PKTS"] + eps)
    ).astype("float32")

    for col in [
        "SRC_TO_DST_SECOND_BYTES",
        "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT",
    ]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32").reset_index(drop=True)
        else:
            out[col] = 0.0

    # TCP flags decoder uses PROTOCOL internally to set IS_TCP.
    # PROTOCOL is not kept as a model feature.
    flags_feat = decode_nf_tcp_flags_0_31(
        df,
        flags_col="TCP_FLAGS",
        proto_col="PROTOCOL",
    ).reset_index(drop=True)

    out = pd.concat([out, flags_feat], axis=1)

    for col in [
        "MIN_TTL",
        "MAX_TTL",
        "MIN_IP_PKT_LEN",
        "MAX_IP_PKT_LEN",
        "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT",
    ]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32").reset_index(drop=True)
        else:
            out[col] = 0.0

    # ---- Labels: one-hot from Attack, benign first
    attack_series = df["Attack"].astype(str).reset_index(drop=True)
    unique_attacks = sorted(attack_series.unique())

    if "benign" in unique_attacks:
        unique_attacks = ["benign"] + [a for a in unique_attacks if a != "benign"]
    else:
        raise ValueError("No 'benign' label found after normalization. Check df['Attack'] values.")

    attack_cat = pd.Categorical(
        attack_series,
        categories=unique_attacks,
        ordered=True,
    )

    attack_ohe = pd.get_dummies(attack_cat).astype(bool)
    label_cols = list(attack_ohe.columns)

    out = pd.concat(
        [out.reset_index(drop=True), attack_ohe.reset_index(drop=True)],
        axis=1,
    )

    with open(os.path.join(out_dir, "attack_labels.json"), "w") as f:
        json.dump(label_cols, f, indent=2)

    # ---- Splits: stratify by Attack
    #
    # IMPORTANT:
    # Do NOT use y_strat.loc[df_test.index].
    # On large parquet files, df may have a non-contiguous index.
    # Instead, split out and y_strat together so the labels stay aligned.
    y_strat = attack_series.reset_index(drop=True)

    df_train, df_temp, y_train, y_temp = train_test_split(
        out,
        y_strat,
        test_size=test_size,
        random_state=random_state,
        stratify=y_strat,
    )

    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        test_size=val_size_within_test,
        random_state=random_state,
        stratify=y_temp,
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # ---- Ensure ports/L7 are not kept
    drop_never_use = [
        c for c in ["L4_SRC_PORT", "L4_DST_PORT", "L7_PROTO"]
        if c in df_train.columns
    ]

    if drop_never_use:
        df_train = df_train.drop(columns=drop_never_use)
        df_val = df_val.drop(columns=drop_never_use, errors="ignore")
        df_test = df_test.drop(columns=drop_never_use, errors="ignore")

    # ---- Scaling: continuous columns only
    scale_cols = [
        "Duration",
        "Packets",
        "Bytes",
        "IN_PKTS",
        "OUT_PKTS",
        "IN_BYTES",
        "OUT_BYTES",
        "SRC_TO_DST_SECOND_BYTES",
        "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT",
        "MIN_TTL",
        "MAX_TTL",
        "MIN_IP_PKT_LEN",
        "MAX_IP_PKT_LEN",
        "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT",
    ]

    scale_cols = [c for c in scale_cols if c in df_train.columns]

    X_train = sanitize_for_scaling(df_train, scale_cols)
    X_val = sanitize_for_scaling(df_val, scale_cols)
    X_test = sanitize_for_scaling(df_test, scale_cols)

    scaler = PowerTransformer()

    df_train.loc[:, scale_cols] = scaler.fit_transform(X_train)
    df_val.loc[:, scale_cols] = scaler.transform(X_val)
    df_test.loc[:, scale_cols] = scaler.transform(X_test)

    # ---- Save parquet splits
    df_train.to_parquet(os.path.join(out_dir, "df_train.parquet"))
    df_val.to_parquet(os.path.join(out_dir, "df_val.parquet"))
    df_test.to_parquet(os.path.join(out_dir, "df_test.parquet"))

    print("✅ Done (NO ports, NO L7).")
    print("Subsampling enabled:", subsample)
    if subsample:
        print("Subsample ratio:", subsample_ratio)
    print("Weekday columns kept:", list(weekday_ohe.columns))
    print("Shapes:", df_train.shape, df_val.shape, df_test.shape)
    print("Num classes:", len(label_cols))
    print("Saved to:", out_dir)

    return df_train, df_val, df_test

# def bi_adj_matrix(batch):
#     edge_index = batch['host', 'to', 'flow'].edge_index  # shape [2, num_edges]
#     num_hosts = batch['host'].num_nodes
#     num_flows = batch['flow'].num_nodes
#
#     # Each edge: edge_index[0][i] = host ID, edge_index[1][i] = flow ID
#     # You might also have edge weights: batch['host', 'to', 'flow'].edge_attr
#     edge_weights = batch['host', 'to', 'flow'].get('edge_attr')
#     if edge_weights is not None:
#         print("Edge weights are not None")
#         weights = edge_weights.view(-1).cpu().numpy()
#     else:
#         print("Edge weights are None")
#         weights = np.ones(edge_index.size(1))
#
#     rows = edge_index[0].cpu().numpy()  # host indices
#     cols = edge_index[1].cpu().numpy()  # flow indices
#
#     W_hf = sp.csr_matrix((weights, (rows, cols)), shape=(num_hosts, num_flows))
#     return W_hf

def corrected_wo_port_preprocess_nf_unsw_nb15_v3_compact(
    raw_path: str,
    out_dir: str = "./NF-UNSW-NB15-v3-compact",
    random_state: int = 0,
    test_size: float = 0.2,
    val_size_within_test: float = 0.5,
    hash_bins: int = 2**16,
    split_mode: str = "random",  # "random" or "time"
):
    """
    Build a compact feature table for NF-UNSW-NB15-v3.

    This version explicitly DOES NOT USE:
      - L4_SRC_PORT, L4_DST_PORT
      - L7_PROTO

    Supported split modes:
      - split_mode="random":
          Stratified random split by Attack.
      - split_mode="time":
          Sort by timestamp and split by contiguous time blocks.
          The first (1 - test_size) fraction becomes train.
          The remaining fraction is split into val/test according to
          val_size_within_test.

    Notes:
      - Scaling/clipping statistics are learned from TRAIN only.
      - Validation and test reuse the train-derived clip bounds.
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PowerTransformer

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(raw_path).copy()

    if "Attack" not in df.columns:
        raise ValueError("This pipeline requires an 'Attack' column for the flow class label.")

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not (0.0 < val_size_within_test < 1.0):
        raise ValueError(f"val_size_within_test must be in (0, 1), got {val_size_within_test}")
    if split_mode not in {"random", "time"}:
        raise ValueError(f"split_mode must be 'random' or 'time', got {split_mode!r}")

    # ---- Required label cleanup
    df["Attack"] = df["Attack"].astype(str).fillna("unknown")
    df.loc[df["Attack"].isin(["nan", "None", ""]), "Attack"] = "unknown"
    df["Attack"] = df["Attack"].str.strip()
    df.loc[df["Attack"].str.lower().isin(["benign", "normal", "background"]), "Attack"] = "benign"

    # ---- Timestamp selection
    start_col = _pick_first_existing(df, ["FLOW_START_MILLISECONDS", "flow_start_milliseconds", "FLOW_START_MS"])
    end_col = _pick_first_existing(df, ["FLOW_END_MILLISECONDS", "flow_end_milliseconds", "FLOW_END_MS"])

    if start_col:
        dt = pd.to_datetime(df[start_col], unit="ms", errors="coerce")
    elif end_col:
        dt = pd.to_datetime(df[end_col], unit="ms", errors="coerce")
    else:
        dt = pd.to_datetime(np.arange(len(df)), unit="s", origin="unix", errors="coerce")

    dt = dt.fillna(pd.Timestamp("1970-01-01"))

    # Keep timestamp around for optional time-based splitting
    df["_split_timestamp"] = dt

    # ---- Timestamp -> weekday/daytime
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_idx = dt.dt.weekday
    weekday_ohe = pd.get_dummies(weekday_idx)
    weekday_ohe.columns = [day_names[i] for i in weekday_ohe.columns]
    weekday_ohe = weekday_ohe.astype(bool)
    weekday_ohe = weekday_ohe.loc[:, weekday_ohe.any(axis=0)]

    daytime = ((dt.dt.second + dt.dt.minute * 60 + dt.dt.hour * 3600) / (24 * 3600)).astype("float32")

    # ---- Duration
    if "FLOW_DURATION_MILLISECONDS" in df.columns:
        duration = (
            pd.to_numeric(df["FLOW_DURATION_MILLISECONDS"], errors="coerce")
            .fillna(0) / 1000.0
        ).astype("float32")
    else:
        duration = pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)

    # ---- Packets / bytes
    in_pkts = (
        pd.to_numeric(df["IN_PKTS"], errors="coerce").fillna(0).astype("float32")
        if "IN_PKTS" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    )
    out_pkts = (
        pd.to_numeric(df["OUT_PKTS"], errors="coerce").fillna(0).astype("float32")
        if "OUT_PKTS" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    )
    packets = (in_pkts + out_pkts).astype("float32")

    in_bytes = (
        pd.to_numeric(df["IN_BYTES"], errors="coerce").fillna(0).astype("float32")
        if "IN_BYTES" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    )
    out_bytes = (
        pd.to_numeric(df["OUT_BYTES"], errors="coerce").fillna(0).astype("float32")
        if "OUT_BYTES" in df.columns else pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    )
    bytes_total = (in_bytes + out_bytes).astype("float32")

    # ---- IP strings
    src_ip_col = _pick_first_existing(df, ["IPV4_SRC_ADDR", "IPv4_SRC_ADDR", "src_ip", "SRC_IP"])
    dst_ip_col = _pick_first_existing(df, ["IPV4_DST_ADDR", "IPv4_DST_ADDR", "dst_ip", "DST_IP"])

    src_ip = df[src_ip_col] if src_ip_col else pd.Series(["0.0.0.0"] * len(df), index=df.index)
    dst_ip = df[dst_ip_col] if dst_ip_col else pd.Series(["0.0.0.0"] * len(df), index=df.index)

    # ---- Build base output
    out = pd.DataFrame(
        {
            "Duration": duration,
            "Src IP Addr": src_ip.astype(str),
            "Dst IP Addr": dst_ip.astype(str),
            "Packets": packets,
            "Bytes": bytes_total,
            "daytime": daytime,
            "_split_timestamp": df["_split_timestamp"],
            "_row_id": np.arange(len(df), dtype=np.int64),
        }
    ).reset_index(drop=True)

    out = pd.concat([out, weekday_ohe.reset_index(drop=True)], axis=1)

    # ---- Compact IP numeric features
    out = pd.concat(
        [
            out,
            _ipv4_features(out["Src IP Addr"], "srcip", hash_bins=hash_bins).reset_index(drop=True),
            _ipv4_features(out["Dst IP Addr"], "dstip", hash_bins=hash_bins).reset_index(drop=True),
        ],
        axis=1,
    )

    # ---- Extra numeric features (NO ports, NO L7)
    out["IN_PKTS"] = in_pkts.reset_index(drop=True)
    out["OUT_PKTS"] = out_pkts.reset_index(drop=True)
    out["IN_BYTES"] = in_bytes.reset_index(drop=True)
    out["OUT_BYTES"] = out_bytes.reset_index(drop=True)

    eps = 1e-6
    out["BYTE_IN_RATIO"] = (out["IN_BYTES"] / (out["IN_BYTES"] + out["OUT_BYTES"] + eps)).astype("float32")
    out["PKT_IN_RATIO"] = (out["IN_PKTS"] / (out["IN_PKTS"] + out["OUT_PKTS"] + eps)).astype("float32")

    for col in [
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"
    ]:
        out[col] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
            if col in df.columns else 0.0
        )

    flags_feat = decode_nf_tcp_flags_0_31(df, flags_col="TCP_FLAGS", proto_col="PROTOCOL").reset_index(drop=True)
    out = pd.concat([out, flags_feat], axis=1)

    for col in ["MIN_TTL", "MAX_TTL", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT"]:
        out[col] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
            if col in df.columns else 0.0
        )

    # ---- Labels
    attack_series = df["Attack"].astype(str)
    unique_attacks = sorted(attack_series.unique())
    if "benign" in unique_attacks:
        unique_attacks = ["benign"] + [a for a in unique_attacks if a != "benign"]
    else:
        raise ValueError("No 'benign' label found after normalization. Check df['Attack'] values.")

    attack_cat = pd.Categorical(attack_series, categories=unique_attacks, ordered=True)
    attack_ohe = pd.get_dummies(attack_cat).astype(bool)
    label_cols = list(attack_ohe.columns)

    out = pd.concat([out.reset_index(drop=True), attack_ohe.reset_index(drop=True)], axis=1)

    with open(os.path.join(out_dir, "attack_labels.json"), "w") as f:
        json.dump(label_cols, f, indent=2)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    if split_mode == "random":
        y_strat = df["Attack"].astype(str)

        df_train, df_temp = train_test_split(
            out,
            test_size=test_size,
            random_state=random_state,
            stratify=y_strat
        )

        y_strat_temp = y_strat.loc[df_temp.index]
        df_val, df_test = train_test_split(
            df_temp,
            test_size=val_size_within_test,
            random_state=random_state,
            stratify=y_strat_temp
        )

    else:  # split_mode == "time"
        out = out.sort_values(["_split_timestamp", "_row_id"], kind="mergesort").reset_index(drop=True)

        n = len(out)
        n_train = int(round(n * (1.0 - test_size)))
        n_train = min(max(n_train, 1), n - 2)

        remaining = n - n_train
        n_val = int(round(remaining * (1.0 - val_size_within_test)))
        n_val = min(max(n_val, 1), remaining - 1)

        df_train = out.iloc[:n_train].copy()
        df_val = out.iloc[n_train:n_train + n_val].copy()
        df_test = out.iloc[n_train + n_val:].copy()

    # Reset after split
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # ---- Train-only clipping + scaling
    scale_cols = [
        "Duration", "Packets", "Bytes",
        "IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES",
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "MIN_TTL", "MAX_TTL",
        "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    ]
    scale_cols = [c for c in scale_cols if c in df_train.columns]

    def _prepare_numeric(df_split: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        X = df_split[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X

    def _clip_with_bounds(df_split: pd.DataFrame, cols: list[str], lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
        X = _prepare_numeric(df_split, cols)
        X = X.clip(lower=lower, upper=upper, axis=1)
        return X

    X_train_raw = _prepare_numeric(df_train, scale_cols)
    upper = X_train_raw.quantile(0.9999)
    lower = X_train_raw.quantile(0.0001)

    X_train = _clip_with_bounds(df_train, scale_cols, lower, upper)
    X_val = _clip_with_bounds(df_val, scale_cols, lower, upper)
    X_test = _clip_with_bounds(df_test, scale_cols, lower, upper)

    scaler = PowerTransformer()
    df_train.loc[:, scale_cols] = scaler.fit_transform(X_train)
    df_val.loc[:, scale_cols] = scaler.transform(X_val)
    df_test.loc[:, scale_cols] = scaler.transform(X_test)

    # ---- Drop internal helper columns before saving
    internal_cols = ["_split_timestamp", "_row_id"]
    df_train = df_train.drop(columns=[c for c in internal_cols if c in df_train.columns])
    df_val = df_val.drop(columns=[c for c in internal_cols if c in df_val.columns])
    df_test = df_test.drop(columns=[c for c in internal_cols if c in df_test.columns])

    # ---- Save
    df_train.to_parquet(os.path.join(out_dir, "df_train.parquet"))
    df_val.to_parquet(os.path.join(out_dir, "df_val.parquet"))
    df_test.to_parquet(os.path.join(out_dir, "df_test.parquet"))

    print(f"✅ Done (NO ports, NO L7). split_mode={split_mode}")
    print("Weekday columns kept:", list(weekday_ohe.columns))
    print("Shapes:", df_train.shape, df_val.shape, df_test.shape)
    print("Num classes:", len(label_cols))
    print("Saved to:", out_dir)

    return df_train, df_val, df_test


if __name__ == "__main__":
    raw_path = "./data/NF-CSE-CIC-IDS2018-v3/NF-CICIDS2018-v3.parquet" # ../data/NF

    # wo_port_preprocess_nf_unsw_nb15_v3_compact(
    #     raw_path=raw_path,
    #     out_dir="./data/NF-UNSW-NB15-v3-compact",
    #     random_state=0,
    # )

    # corrected_wo_port_preprocess_nf_unsw_nb15_v3_compact(
    #     raw_path=raw_path,
    #     split_mode="random",
    #     test_size=0.2,
    #     val_size_within_test=0.5,
    #
    # )

    wo_port_preprocess_nf_unsw_nb15_v3_compact(
        raw_path=raw_path,
        out_dir='./data/NF-CSE-CIC-IDS2018-v3',
        random_state=0,
        test_size=0.2,
        subsample=True,
        subsample_ratio=0.2
    )
