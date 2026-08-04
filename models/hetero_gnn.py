import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv, SAGEConv, GATConv, HGTConv


class HeteroGNN(nn.Module):
    """
    Copied from the source: https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python/blob/main/Chapter16/chapter16.ipynb

    """

    def __init__(self, dim_h, dim_out, num_layers):
        super(HeteroGNN, self).__init__()

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('host', 'to', 'flow'): SAGEConv((-1, -1), dim_h),
                ('flow', 'to', 'host'): SAGEConv((-1, -1), dim_h),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(dim_h, dim_out)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['flow'])

    def get_flow_embeddings(self, batch):
        """
                Returns the intermediate flow node embeddings before classification.
                Used for Mahalanobis open-set detection.
                """
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        flow_emb = x_dict["flow"]
        return flow_emb


class HeteroGAT(nn.Module):
    def __init__(self, dim_h, dim_out, num_layers, num_heads, dropout=0.3):
        super(HeteroGAT, self).__init__()

        # Input projection for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type in ['host', 'flow']:
            self.lin_dict[node_type] = Linear(-1, dim_h)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # For hidden layers: concat heads, for last layer: average with 1 head
            if i < num_layers - 1:
                conv = HeteroConv({
                    ('host', 'to', 'flow'): GATConv(dim_h, dim_h // num_heads,
                                                    heads=num_heads,
                                                    add_self_loops=False,
                                                    concat=True,
                                                    dropout=dropout),
                    ('flow', 'to', 'host'): GATConv(dim_h, dim_h // num_heads,
                                                    heads=num_heads,
                                                    add_self_loops=False,
                                                    concat=True,
                                                    dropout=dropout),
                }, aggr='sum')
            else:
                # Last layer: single head or average
                conv = HeteroConv({
                    ('host', 'to', 'flow'): GATConv(dim_h, dim_h,
                                                    heads=1,
                                                    add_self_loops=False,
                                                    concat=False,
                                                    dropout=dropout),
                    ('flow', 'to', 'host'): GATConv(dim_h, dim_h,
                                                    heads=1,
                                                    add_self_loops=False,
                                                    concat=False,
                                                    dropout=dropout),
                }, aggr='sum')

            self.convs.append(conv)

            # Layer normalization for stability
            norm_dict = nn.ModuleDict()
            for node_type in ['host', 'flow']:
                norm_dict[node_type] = nn.LayerNorm(dim_h)
            self.norms.append(norm_dict)

        self.lin = Linear(dim_h, dim_out)
        self.dropout = dropout

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        # Input projection
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        # GAT layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_dict_out = conv(x_dict, edge_index_dict)

            # Residual connection + normalization
            for key in x_dict_out.keys():
                x_dict_out[key] = norm[key](x_dict_out[key] + x_dict[key])
                x_dict_out[key] = F.elu(x_dict_out[key])
                x_dict_out[key] = F.dropout(x_dict_out[key], p=self.dropout, training=self.training)

            x_dict = x_dict_out

        return self.lin(x_dict['flow'])

    def get_flow_embeddings(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        # Input projection
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        # GAT layers
        for conv, norm in zip(self.convs, self.norms):
            x_dict_out = conv(x_dict, edge_index_dict)

            for key in x_dict_out.keys():
                x_dict_out[key] = norm[key](x_dict_out[key] + x_dict[key])
                x_dict_out[key] = F.elu(x_dict_out[key])
                x_dict_out[key] = F.dropout(x_dict_out[key], p=self.dropout, training=self.training)

            x_dict = x_dict_out

        return x_dict['flow']


# class HeteroGAT(nn.Module):
#     def __init__(self, dim_h, dim_out, num_layers, num_heads):
#         super(HeteroGAT, self).__init__()
#
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HeteroConv({
#                 ('host', 'to', 'flow'): GATConv((-1, -1), dim_h, heads=num_heads, add_self_loops=False, concat=False),
#                 ('flow', 'to', 'host'): GATConv((-1, -1), dim_h, heads=num_heads, add_self_loops=False, concat=False),
#             }, aggr='sum')
#             self.convs.append(conv)
#
#         self.lin = Linear(dim_h, dim_out)
#
#     def forward(self, batch):
#         x_dict = batch.x_dict
#         edge_index_dict = batch.edge_index_dict
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
#         return self.lin(x_dict['flow'])
#
#     def get_flow_embeddings(self, batch):
#         """
#                 Returns the intermediate flow node embeddings before classification.
#                 Used for Mahalanobis open-set detection.
#                 """
#         x_dict = batch.x_dict
#         edge_index_dict = batch.edge_index_dict
#
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
#
#         flow_emb = x_dict["flow"]
#         return flow_emb

# class HGT(nn.Module):
#
#     def __init__(self, dim_h, dim_out, num_layers, num_heads, data):
#         super(HGT, self).__init__()
#
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HGTConv(dim_h, dim_h, data.metadata(),
#                            num_heads, group='sum')
#             self.convs.append(conv)
#
#         self.lin = Linear(dim_h, dim_out)
#
#     def forward(self, batch):
#         x_dict = batch.x_dict
#         edge_index_dict = batch.edge_index_dict
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
#         return self.lin(x_dict['flow'])


class HGT(nn.Module):
    def __init__(self, dim_h, dim_out, num_layers, num_heads, metadata):
        super().__init__()

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, dim_h)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(dim_h, dim_h, metadata,
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(dim_h, dim_out)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['flow'])

    def get_flow_embeddings(self, batch):
        """
                Returns the intermediate flow node embeddings before classification.
                Used for Mahalanobis open-set detection.
                """
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict["flow"]
