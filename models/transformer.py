import torch.nn as nn


class TransformerClassifier(nn.Module):
    """
    Redundant because data does not include any sequence. It is an overkill to use Transformers for this.
    MLP is much better
    """

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        data = batch.x_dict['flow']
        x = data.x if hasattr(data, 'x') else data

        if x.dim() == 2:
           x = x.unsqueeze(1)  # Add time dimension if needed

        x = self.input_proj(x)
        x = self.transformer(x)
        return self.classifier(x[:, -1])  # use last token
