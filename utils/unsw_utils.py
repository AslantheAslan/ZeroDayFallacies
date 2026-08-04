import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, LabelEncoder


def _pick_first_existing(df: pd.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _to_num(s, default=0):
    if isinstance(s, (int, float, np.number)):
        return pd.Series([s])
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _safe_uint8(series: pd.Series, default=0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(default)
    x = x.clip(0, 255).astype("int32")
    return x.astype("uint8")


def _ipv4_features(ip_series: pd.Series, prefix: str, hash_bins: int = 2**16) -> pd.DataFrame:
    """
    Convert IPv4 string to compact features:
      - {prefix}_o1..o4  (octets)
      - {prefix}_int32   (packed 32-bit integer stored as int64)
      - {prefix}_hash    (bucketed stable hash)

    Invalid / non-4-octet / out-of-range => ALL ZEROS.
    Uses NumPy arrays for robust bit ops.
    """
    ip = ip_series.astype(str)

    # Must be exactly 4 octets
    mask = ip.str.match(r"^\d{1,3}(\.\d{1,3}){3}$", na=False)
    ip = ip.where(mask, other="0.0.0.0")

    parts = ip.str.split(".", expand=True)

    # Convert each octet to numpy int64 safely
    def octet(col):
        arr = pd.to_numeric(col, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        return arr

    o1 = octet(parts[0])
    o2 = octet(parts[1])
    o3 = octet(parts[2])
    o4 = octet(parts[3])

    # Range check; anything invalid => zero out all octets (per your requirement)
    valid = (
        (o1 >= 0) & (o1 <= 255) &
        (o2 >= 0) & (o2 <= 255) &
        (o3 >= 0) & (o3 <= 255) &
        (o4 >= 0) & (o4 <= 255)
    )
    o1 = np.where(valid, o1, 0)
    o2 = np.where(valid, o2, 0)
    o3 = np.where(valid, o3, 0)
    o4 = np.where(valid, o4, 0)

    # Pack into 32-bit integer (kept as int64 to avoid overflow surprises)
    ip_int32 = ((o1 << 24) | (o2 << 16) | (o3 << 8) | o4).astype(np.int64)

    # Stable bucket hash from int32 values (no giant cardinality)
    # Using pandas hashing on a Series is fine, but we feed it clean int64.
    ip_hash = (pd.util.hash_pandas_object(pd.Series(ip_int32), index=False)
               .astype("uint64") % np.uint64(hash_bins)).astype("int32").to_numpy()

    return pd.DataFrame({
        f"{prefix}_o1": o1.astype(np.int16),
        f"{prefix}_o2": o2.astype(np.int16),
        f"{prefix}_o3": o3.astype(np.int16),
        f"{prefix}_o4": o4.astype(np.int16),

    })


def decode_nf_tcp_flags_0_31(df: pd.DataFrame, flags_col: str = "TCP_FLAGS",
                             proto_col: str = "PROTOCOL") -> pd.DataFrame:
    """
    NF-UNSW-NB15-v3 TCP_FLAGS observed in [0..31] => 5-bit cumulative mask:
      FIN=1, SYN=2, RST=4, PSH=8, ACK=16

    IMPORTANT:
      - It's cumulative over the flow (so combos like 19, 27 are normal).
      - For non-TCP flows (PROTOCOL != 6) set all outputs to 0 (prevents mixing).
    """
    f = pd.to_numeric(df.get(flags_col, 0), errors="coerce").fillna(0).astype(np.int64).clip(0, 31)
    proto = pd.to_numeric(df.get(proto_col, 0), errors="coerce").fillna(0).astype(np.int64)
    is_tcp = (proto == 6).astype("int32")

    FIN = (((f & 1) != 0) & (is_tcp == 1)).astype("int32")
    SYN = (((f & 2) != 0) & (is_tcp == 1)).astype("int32")
    RST = (((f & 4) != 0) & (is_tcp == 1)).astype("int32")
    PSH = (((f & 8) != 0) & (is_tcp == 1)).astype("int32")
    ACK = (((f & 16) != 0) & (is_tcp == 1)).astype("int32")

    return pd.DataFrame(
        {
            "IS_TCP": is_tcp,
            "FIN": FIN,
            "SYN": SYN,
            "RST": RST,
            "PSH": PSH,
            "ACK": ACK,
            "TCP_FLAG_COUNT": (FIN + SYN + RST + PSH + ACK).astype("int32"),
            "TCP_FLAGS_RAW": f.astype("int32"),  # optional but often useful
        }
    )

def sanitize_for_scaling(df: pd.DataFrame, cols: list[str], clip_quantile: float = 0.9999) -> pd.DataFrame:
    """
    - coerces to float64
    - replaces inf/-inf with NaN
    - fills NaN with 0
    - clips extreme outliers to a high quantile (prevents overflow / huge floats)
    """
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # clip each column to a robust upper bound
    # (quantile computed on training data only)
    upper = X.quantile(clip_quantile)
    lower = X.quantile(1 - clip_quantile)

    X = X.clip(lower=lower, upper=upper, axis=1)
    return X
