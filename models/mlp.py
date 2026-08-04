import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout=0.3):
        super(MLP, self).__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        self.layers = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        # Extract flow node features from batch
        if hasattr(batch, 'x_dict'):
            # batch is a heterogeneous graph batch
            x = batch.x_dict['flow']
        else:
            # batch is already a tensor
            x = batch

        for layer in self.layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def get_flow_embeddings(self, batch):
        # Extract flow node features from batch
        if hasattr(batch, 'x_dict'):
            x = batch.x_dict['flow']
        else:
            x = batch

        for layer in self.layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class meta_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout=0.5):
        super(meta_MLP, self).__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        self.layers = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        # x_dict = batch.x_dict
        data = batch['flow']
        emb_raw, emb, cos, delta = data.x_emb, data.x_emb_raw, data.x_cos, data.x_delta
        inp = torch.cat([emb_raw, emb, cos, delta], dim=1)

        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)
        out = self.classifier(inp)
        return out

    def get_flow_embeddings(self, batch):
        x_dict = batch.x_dict
        data = x_dict['flow']
        x, emb_raw, emb, cos, delta = data.x, data.x_emb_raw, data.x_emb, data.x_cos, data.x_delta
        inp = torch.cat([emb_raw, emb, cos, delta], dim=1)

        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)

        return inp


class GatedFusionMetaMLP(torch.nn.Module):
    def __init__(self, emb_raw_dim, emb_dim, hidden_dim, num_classes, num_layers,
                 dropout=0.5, fusion_type="simple", gate_hidden_dim=64):
        super(GatedFusionMetaMLP, self).__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        self.fusion_type = fusion_type
        self.dropout = dropout

        # Gated fusion component
        self.fusion_dim = max(emb_raw_dim, emb_dim)

        # Project embeddings to same dimension
        self.proj_emb_raw = nn.Linear(emb_raw_dim, self.fusion_dim)
        self.proj_emb = nn.Linear(emb_dim, self.fusion_dim)

        # Gate network
        if fusion_type == "simple":
            # Single gate for entire vector
            self.gate = nn.Sequential(
                nn.Linear(emb_raw_dim + emb_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid()
            )
        elif fusion_type == "elementwise":
            # Element-wise gating
            self.gate = nn.Sequential(
                nn.Linear(emb_raw_dim + emb_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, self.fusion_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError("fusion_type must be 'simple' or 'elementwise'")

        # MLP layers (same structure as original)
        self.layers = nn.ModuleList()

        # First layer (input is now fusion_dim instead of concatenated embeddings)
        self.layers.append(nn.Linear(self.fusion_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layers
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Anomaly detection head (optional)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def gated_fusion(self, emb_raw, emb):
        """Apply gated fusion to combine embeddings"""
        # Project to same dimension
        proj_emb_raw = self.proj_emb_raw(emb_raw)
        proj_emb = self.proj_emb(emb)

        # Compute gate values
        concat_input = torch.cat([emb_raw, emb], dim=1)
        gate_values = self.gate(concat_input)

        # Apply gating
        if self.fusion_type == "simple":
            # Single gate value for entire vector
            fused_emb = gate_values * proj_emb_raw + (1 - gate_values) * proj_emb
        else:  # elementwise
            # Element-wise gating
            fused_emb = gate_values * proj_emb_raw + (1 - gate_values) * proj_emb

        return fused_emb, gate_values

    def forward(self, batch):
        data = batch['flow']
        emb_raw, emb = data.x_emb_raw, data.x_emb

        # Gated fusion
        fused_emb, gate_values = self.gated_fusion(emb_raw, emb)

        # Pass through MLP layers
        inp = fused_emb
        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)

        # Classification output
        class_out = self.classifier(inp)

        return class_out

    def forward_with_anomaly(self, batch):
        """Forward pass that also returns anomaly scores"""
        data = batch['flow']
        emb_raw, emb = data.x_emb_raw, data.x_emb

        # Gated fusion
        fused_emb, gate_values = self.gated_fusion(emb_raw, emb)

        # Pass through MLP layers
        inp = fused_emb
        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)

        # Classification and anomaly outputs
        class_out = self.classifier(inp)
        anomaly_score = self.anomaly_head(inp)

        return class_out, anomaly_score, gate_values

    def get_flow_embeddings(self, batch):
        """Get intermediate embeddings (matching original interface)"""
        data = batch['flow'] if 'flow' in batch else batch.x_dict['flow']
        emb_raw, emb = data.x_emb_raw, data.x_emb

        # Gated fusion
        fused_emb, _ = self.gated_fusion(emb_raw, emb)

        # Pass through MLP layers
        inp = fused_emb
        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)

        return inp

    def get_gate_weights(self, batch):
        """Get gate weights for interpretability"""
        data = batch['flow']
        emb_raw, emb = data.x_emb_raw, data.x_emb

        _, gate_values = self.gated_fusion(emb_raw, emb)
        return gate_values


class ExtendedGatedFusionMetaMLP(torch.nn.Module):
    def __init__(self, emb_raw_dim, emb_dim, cos_dim, delta_dim, hidden_dim,
                 num_classes, num_layers, dropout=0.5):
        super(ExtendedGatedFusionMetaMLP, self).__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        self.dropout = dropout

        # Determine fusion dimension
        self.fusion_dim = max(emb_raw_dim, emb_dim, cos_dim, delta_dim)

        # Project all inputs to same dimension
        self.proj_emb_raw = nn.Linear(emb_raw_dim, self.fusion_dim)
        self.proj_emb = nn.Linear(emb_dim, self.fusion_dim)
        self.proj_cos = nn.Linear(cos_dim, self.fusion_dim)
        self.proj_delta = nn.Linear(delta_dim, self.fusion_dim)

        # Multi-input attention gate
        total_input_dim = emb_raw_dim + emb_dim + cos_dim + delta_dim
        self.attention_gate = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 weights for 4 inputs
            nn.Softmax(dim=1)
        )

        # MLP layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(self.fusion_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layers
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def multi_gated_fusion(self, emb_raw, emb, cos, delta):
        """Apply attention-based fusion to all inputs"""
        # Project to same dimension
        proj_emb_raw = self.proj_emb_raw(emb_raw)
        proj_emb = self.proj_emb(emb)
        proj_cos = self.proj_cos(cos)
        proj_delta = self.proj_delta(delta)

        # Compute attention weights
        concat_input = torch.cat([emb_raw, emb, cos, delta], dim=1)
        attention_weights = self.attention_gate(concat_input)  # [batch_size, 4]

        # Weighted fusion
        fused_emb = (attention_weights[:, 0:1] * proj_emb_raw +
                     attention_weights[:, 1:2] * proj_emb +
                     attention_weights[:, 2:3] * proj_cos +
                     attention_weights[:, 3:4] * proj_delta)

        return fused_emb, attention_weights

    def forward(self, batch):
        data = batch['flow']
        emb_raw, emb, cos, delta = data.x_emb_raw, data.x_emb, data.x_cos, data.x_delta

        # Multi-input gated fusion
        fused_emb, attention_weights = self.multi_gated_fusion(emb_raw, emb, cos, delta)

        # Pass through MLP layers
        inp = fused_emb
        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)

        class_out = self.classifier(inp)
        return class_out

    def forward_with_anomaly(self, batch):
        """Forward pass that also returns anomaly scores and attention weights"""
        data = batch['flow']
        emb_raw, emb, cos, delta = data.x_emb_raw, data.x_emb, data.x_cos, data.x_delta

        # Multi-input gated fusion
        fused_emb, attention_weights = self.multi_gated_fusion(emb_raw, emb, cos, delta)

        # Pass through MLP layers
        inp = fused_emb
        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)

        class_out = self.classifier(inp)
        anomaly_score = self.anomaly_head(inp)

        return class_out, anomaly_score, attention_weights

    def get_flow_embeddings(self, batch):
        """Get intermediate embeddings"""
        data = batch['flow'] if 'flow' in batch else batch.x_dict['flow']
        emb_raw, emb, cos, delta = data.x_emb_raw, data.x_emb, data.x_cos, data.x_delta

        fused_emb, _ = self.multi_gated_fusion(emb_raw, emb, cos, delta)

        inp = fused_emb
        for layer in self.layers:
            inp = F.relu(layer(inp))
            inp = F.dropout(inp, p=self.dropout, training=self.training)

        return inp
