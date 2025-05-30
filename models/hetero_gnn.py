import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv, SAGEConv


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