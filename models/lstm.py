import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        x_dict = batch.x_dict
        data = x_dict['flow']
        x = data.x if hasattr(data, 'x') else data

        if x.dim() == 2:
           x = x.unsqueeze(1)  # Add time dimension if needed

        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])
