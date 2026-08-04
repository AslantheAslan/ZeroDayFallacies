import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, DataLoader

import pandas as pd
import scipy.sparse as sp
from typing import List, Tuple

from joblib import Parallel, delayed

# !!!!get_connections and create_dataloader funcs were implemented like below before
#
def get_connections(ip_map, src_ip, dst_ip):
    # Convert IPs to indices using the mapping
    src1 = np.array([ip_map[ip] for ip in src_ip])
    src2 = np.array([ip_map[ip] for ip in dst_ip])

    # Create source indices array
    src = np.column_stack((src1, src2)).flatten()

    # Create destination indices array
    dst = np.arange(len(src_ip))
    dst = np.column_stack((dst, dst)).flatten()

    # Convert to tensors efficiently
    edge_index_1 = torch.tensor(np.array([src, dst]), dtype=torch.int32)
    edge_index_2 = torch.tensor(np.array([dst, src]), dtype=torch.int32)
    return edge_index_1, edge_index_2


def create_dataloader(df, labels, BATCH_SIZE, features_flow, features_host, subgraph_size=1024, shuffle=False, pre_transform=None):
    data = []
    # labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']
    n_subgraphs = len(df) // subgraph_size

    # First, ensure all features_flow columns are float
    numerical_df = df.copy()
    for col in features_flow:
        if numerical_df[col].dtype == 'object':
            # Convert string columns to float
            numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce')
        # Ensure all columns are float64
        numerical_df[col] = numerical_df[col].astype('float64')

    for i in range(1, n_subgraphs + 1):
        subgraph = numerical_df[(i - 1) * subgraph_size:i * subgraph_size]

        src_ip = subgraph['Src IP Addr'].to_numpy()
        dst_ip = subgraph['Dst IP Addr'].to_numpy()

        ip_map = {ip: index for index, ip in enumerate(np.unique(np.append(src_ip, dst_ip)))}
        host_to_flow, flow_to_host = get_connections(ip_map, src_ip, dst_ip)

        batch = HeteroData()
        # !!! HOST FEATURES WERE LIKE BELOW ORIGINALLY !!!
        batch['host'].x = torch.Tensor(subgraph[features_host].to_numpy()).float()

        # batch['host'].x = torch.randint(low=0, high=2, size=(len(ip_map), len(features_host)), dtype=torch.float)
        # batch['host'].x = torch.ones((len(ip_map), len(features_host)), dtype=torch.float)
        # batch['host'].x = torch.zeros((len(ip_map), len(features_host)), dtype=torch.float)

        # Convert features_flow to tensor after ensuring all values are float
        flow_features = subgraph[features_flow].to_numpy()
        batch['flow'].x = torch.FloatTensor(flow_features)

        batch['flow'].y = torch.Tensor(subgraph[labels].to_numpy()).float()
        batch['host', 'flow'].edge_index = host_to_flow.long()  # to long() for RemoveIsolatedNodes transformation
        batch['flow', 'host'].edge_index = flow_to_host.long()

        if pre_transform is not None:
            batch = pre_transform(batch)

        data.append(batch)

    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=True)

# def get_connections(ip_map, src_ip, dst_ip):
#     """
#     Build bipartite edges:
#       host(src_ip) -> flow
#       host(dst_ip) -> flow
#     and reverse edges flow -> host.
#     """
#     src_host_idx = np.array([ip_map[ip] for ip in src_ip], dtype=np.int64)
#     dst_host_idx = np.array([ip_map[ip] for ip in dst_ip], dtype=np.int64)
#
#     host_indices = np.column_stack((src_host_idx, dst_host_idx)).reshape(-1)
#     flow_indices = np.arange(len(src_ip), dtype=np.int64)
#     flow_indices = np.column_stack((flow_indices, flow_indices)).reshape(-1)
#
#     edge_index_hf = torch.tensor(
#         np.vstack([host_indices, flow_indices]),
#         dtype=torch.long
#     )
#     edge_index_fh = torch.tensor(
#         np.vstack([flow_indices, host_indices]),
#         dtype=torch.long
#     )
#     return edge_index_hf, edge_index_fh
#
#
# def _safe_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
#     """
#     Convert columns to numeric float64 safely.
#     Non-numeric values become 0.
#     """
#     out = df[cols].copy()
#     for col in cols:
#         if out[col].dtype == "object":
#             out[col] = pd.to_numeric(out[col], errors="coerce")
#         out[col] = out[col].astype("float64")
#     out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
#     return out
#
#
# def _extract_single_host_features(ip: str, subgraph: pd.DataFrame, features_host: list[str]) -> np.ndarray:
#     """
#     Create ONE host feature row for ONE unique IP.
#
#     Strategy:
#     - If the IP appears as source, use the src-side host features from the first such row.
#     - Else if it appears as destination, use the dst-side host features from the first such row.
#     - Else return zeros (should almost never happen).
#
#     This assumes features_host contains host-side IP-derived columns, e.g.
#       CIDDS:  ipsrc_* + ipdst_*
#       NF-UNSW: srcip_* + dstip_*
#
#     We map them to a canonical host representation by taking the relevant half only.
#     """
#     # Try to infer src-side and dst-side host columns
#     src_cols = [c for c in features_host if c.startswith("ipsrc_") or c.startswith("srcip_")]
#     dst_cols = [c for c in features_host if c.startswith("ipdst_") or c.startswith("dstip_")]
#
#     # Fallback: if we cannot split them, use the first matching row as-is
#     if len(src_cols) + len(dst_cols) != len(features_host):
#         rows = subgraph[(subgraph["Src IP Addr"] == ip) | (subgraph["Dst IP Addr"] == ip)]
#         if not rows.empty:
#             vals = pd.to_numeric(rows.iloc[0][features_host], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
#             return vals
#         return np.zeros(len(features_host), dtype=np.float32)
#
#     # Canonical host feature layout keeps same dimensionality as before:
#     # [src_side_features, dst_side_features]
#     # For a unique host, we fill BOTH halves with the host's own endpoint features.
#     src_rows = subgraph[subgraph["Src IP Addr"] == ip]
#     if not src_rows.empty:
#         src_vals = pd.to_numeric(src_rows.iloc[0][src_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
#         if len(dst_cols) == len(src_cols):
#             return np.concatenate([src_vals, src_vals]).astype(np.float32)
#         else:
#             # uneven split fallback
#             dst_pad = np.zeros(len(dst_cols), dtype=np.float32)
#             return np.concatenate([src_vals, dst_pad]).astype(np.float32)
#
#     dst_rows = subgraph[subgraph["Dst IP Addr"] == ip]
#     if not dst_rows.empty:
#         dst_vals = pd.to_numeric(dst_rows.iloc[0][dst_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
#         if len(src_cols) == len(dst_cols):
#             return np.concatenate([dst_vals, dst_vals]).astype(np.float32)
#         else:
#             src_pad = np.zeros(len(src_cols), dtype=np.float32)
#             return np.concatenate([src_pad, dst_vals]).astype(np.float32)
#
#     return np.zeros(len(features_host), dtype=np.float32)
#
#
# def _build_host_feature_matrix(subgraph: pd.DataFrame, all_ips: np.ndarray, features_host: list[str]) -> np.ndarray:
#     """
#     Build host.x with exactly one row per unique IP, in the same order as all_ips.
#     """
#     host_rows = [
#         _extract_single_host_features(ip, subgraph, features_host)
#         for ip in all_ips
#     ]
#     return np.stack(host_rows).astype(np.float32)
#
#
# def create_dataloader(
#     df,
#     labels,
#     BATCH_SIZE,
#     features_flow,
#     features_host,
#     subgraph_size=1024,
#     shuffle=False,
#     pre_transform=None
# ):
#     data = []
#
#     # Ensure flow features are numeric
#     numerical_df = df.copy()
#     numerical_df[features_flow] = _safe_numeric_frame(numerical_df, features_flow)
#
#     # Labels should be numeric too
#     numerical_df[labels] = numerical_df[labels].apply(pd.to_numeric, errors="coerce").fillna(0)
#
#     n_rows = len(numerical_df)
#     if n_rows == 0:
#         return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=True)
#
#     # Keep remainder instead of dropping it
#     for start in range(0, n_rows, subgraph_size):
#         end = min(start + subgraph_size, n_rows)
#         subgraph = numerical_df.iloc[start:end].copy()
#
#         if len(subgraph) == 0:
#             continue
#
#         src_ip = subgraph["Src IP Addr"].astype(str).to_numpy()
#         dst_ip = subgraph["Dst IP Addr"].astype(str).to_numpy()
#
#         # Unique IPs define host nodes
#         all_ips = np.unique(np.concatenate([src_ip, dst_ip]))
#         ip_map = {ip: index for index, ip in enumerate(all_ips)}
#
#         host_to_flow, flow_to_host = get_connections(ip_map, src_ip, dst_ip)
#
#         # Correct host feature matrix: one row per unique host
#         host_features = _build_host_feature_matrix(subgraph, all_ips, features_host)
#
#         # Flow features and labels
#         flow_features = subgraph[features_flow].to_numpy(dtype=np.float32)
#         flow_labels = subgraph[labels].to_numpy(dtype=np.float32)
#
#         batch = HeteroData()
#         batch["host"].x = torch.tensor(host_features, dtype=torch.float32)
#         batch["flow"].x = torch.tensor(flow_features, dtype=torch.float32)
#         batch["flow"].y = torch.tensor(flow_labels, dtype=torch.float32)
#         batch["host", "flow"].edge_index = host_to_flow
#         batch["flow", "host"].edge_index = flow_to_host
#
#         if pre_transform is not None:
#             batch = pre_transform(batch)
#
#         data.append(batch)
#
#     return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=True)

class GaussianNoise:
    """Adds Gaussian noise to node features for augmentation."""
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for node_type in data.node_types:
            if 'x' in data[node_type]:
                noise = torch.randn_like(data[node_type].x) * self.std + self.mean
                data[node_type].x = data[node_type].x + noise
        return data

def build_bipartite_adjacency(
        host_to_flow: torch.Tensor,
        flow_to_host: torch.Tensor,
        num_hosts: int,
        num_flows: int,
        edge_weight: float = 1.0):
    """
    Builds a bipartite adjacency matrix W_hf (hosts x flows) from PyTorch edge_index tensors.

    Args:
        host_to_flow (Tensor): Edge index tensor from host to flow, shape (2, num_edges)
        flow_to_host (Tensor): Edge index tensor from flow to host, shape (2, num_edges)
        edge_weight (float): Scalar weight assigned to each edge (default: 1.0)

    Returns:
        W_hf (csr_matrix): Sparse matrix of shape (num_hosts, num_flows)
        num_hosts (int): Number of host nodes
        num_flows (int): Number of flow nodes
    """
    # host → flow edges
    rows_hf = host_to_flow[0].cpu().numpy()
    cols_hf = host_to_flow[1].cpu().numpy()

    # flow → host edges (flip to host → flow)
    rows_fh = flow_to_host[1].cpu().numpy()
    cols_fh = flow_to_host[0].cpu().numpy()

    # Combine all edges
    all_rows = np.concatenate([rows_hf, rows_fh])  # hosts
    all_cols = np.concatenate([cols_hf, cols_fh])  # flows

    # Build sparse matrix
    weights = np.full_like(all_rows, edge_weight, dtype=np.float32)
    W_hf = sp.csr_matrix((weights, (all_rows, all_cols)), shape=(num_hosts, num_flows))

    return W_hf, num_hosts, num_flows


# def create_full_graph(df, features_host: List[str], features_flow: List[str], label_cols: List[str]) -> HeteroData:
#     # Ensure flow features are numerical
#     numerical_df = df.copy()
#     for col in features_flow:
#         if numerical_df[col].dtype == 'object':
#             numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce')
#         numerical_df[col] = numerical_df[col].astype('float64')
#
#     # Extract IPs
#     src_ip = numerical_df['Src IP Addr'].to_numpy()
#     dst_ip = numerical_df['Dst IP Addr'].to_numpy()
#
#     # Unique hosts
#     ip_map = {ip: i for i, ip in enumerate(np.unique(np.concatenate([src_ip, dst_ip])))}
#
#     # Edge indices
#     host_to_flow, flow_to_host = get_connections(ip_map, src_ip, dst_ip)
#
#     # Node features
#     host_features = torch.tensor(numerical_df[features_host].drop_duplicates().to_numpy(), dtype=torch.float)
#     flow_features = torch.tensor(numerical_df[features_flow].to_numpy(), dtype=torch.float)
#     flow_labels = torch.tensor(numerical_df[label_cols].to_numpy(), dtype=torch.float)
#
#     # Build graph
#     data = HeteroData()
#     data['host'].x = host_features
#     data['flow'].x = flow_features
#     data['flow'].y = flow_labels
#     data['host', 'flow'].edge_index = host_to_flow
#     data['flow', 'host'].edge_index = flow_to_host
#
#     return data

def extract_host_row(ip, df, features_host):
    rows = df[(df['Src IP Addr'] == ip) | (df['Dst IP Addr'] == ip)]
    if not rows.empty:
        return rows.iloc[0][features_host].to_numpy()
    else:
        return np.zeros(len(features_host), dtype=np.float32)

def create_full_graph(
    df: pd.DataFrame,
    features_host: List[str],
    features_flow: List[str],
    label_cols: List[str],
    n_jobs: int = -1  # Use all available CPU cores
) -> HeteroData:
    import torch
    from torch_geometric.data import HeteroData

    numerical_df = df.copy()

    for col in features_flow:
        if numerical_df[col].dtype == 'object':
            numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce')
        numerical_df[col] = numerical_df[col].astype('float64')

    src_ip = numerical_df['Src IP Addr'].to_numpy()
    dst_ip = numerical_df['Dst IP Addr'].to_numpy()
    all_ips = np.unique(np.concatenate([src_ip, dst_ip]))
    ip_map = {ip: idx for idx, ip in enumerate(all_ips)}

    # ✅ Parallel host feature extraction
    host_feature_rows = Parallel(n_jobs=n_jobs)(
        delayed(extract_host_row)(ip, numerical_df, features_host)
        for ip in all_ips
    )
    host_features = torch.tensor(np.stack(host_feature_rows), dtype=torch.float)

    # Flow features and labels
    flow_features = torch.tensor(numerical_df[features_flow].to_numpy(), dtype=torch.float)
    flow_labels = torch.tensor(numerical_df[label_cols].to_numpy(), dtype=torch.float)

    # Edges
    host_to_flow, flow_to_host = get_connections(ip_map, src_ip, dst_ip)

    # HeteroData object
    data = HeteroData()
    data['host'].x = host_features
    data['flow'].x = flow_features
    data['flow'].y = flow_labels
    data['host', 'flow'].edge_index = host_to_flow
    data['flow', 'host'].edge_index = flow_to_host

    return data