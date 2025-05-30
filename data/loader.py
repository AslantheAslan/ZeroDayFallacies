import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, DataLoader

import pandas as pd


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


def create_dataloader(df, labels, BATCH_SIZE, features_flow, features_host, subgraph_size=1024):
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
        batch['host'].x = torch.Tensor(subgraph[features_host].to_numpy()).float()

        # Convert features_flow to tensor after ensuring all values are float
        flow_features = subgraph[features_flow].to_numpy()
        batch['flow'].x = torch.FloatTensor(flow_features)

        batch['flow'].y = torch.Tensor(subgraph[labels].to_numpy()).float()
        batch['host', 'flow'].edge_index = host_to_flow
        batch['flow', 'host'].edge_index = flow_to_host
        data.append(batch)

    return DataLoader(data, batch_size=BATCH_SIZE)
