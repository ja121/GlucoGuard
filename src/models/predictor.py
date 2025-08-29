import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from src.models.building_blocks import MultiHeadTemporalAttention, GatedResidualNetwork
from src.data_processing.cgm_dataset import CGMConfig

class TemporalBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.norm1 = nn.LayerNorm(d_model)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.norm2 = nn.LayerNorm(d_model)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x_res = x
        x = x.permute(0, 2, 1) # (batch, d_model, seq_len)
        x = self.dropout1(self.relu1(self.norm1(self.conv1(x).permute(0, 2, 1))))
        x = x.permute(0, 2, 1) # (batch, d_model, seq_len)
        x = self.dropout2(self.relu2(self.norm2(self.conv2(x).permute(0, 2, 1))))
        return x + x_res

class HierarchicalGlucosePredictor(nn.Module):
    def __init__(self, config: CGMConfig, n_features=50):
        super().__init__()
        self.config = config

        # Input embedding
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(config.sequence_length, 128)

        # Multi-scale temporal blocks
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(d_model=128, kernel_size=k, dilation=d)
            for k, d in [(3, 1), (3, 2), (3, 4), (5, 1), (5, 2)]
        ])

        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadTemporalAttention(d_model=128, n_heads=8)
            for _ in range(3)
        ])

        # GRN for feature selection
        self.grn = GatedResidualNetwork(d_model=128, d_hidden=256)

        # Multi-task heads
        self.glucose_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Predict 30min, 1hr, 1.5hr
        )

        self.risk_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 6)  # Hypo/Hyper for 3 time horizons
        )

        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, 3)  # Uncertainty for each prediction
        )

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x, return_attention=False):
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1)]

        # Multi-scale temporal processing
        temporal_features = []
        for block in self.temporal_blocks:
            temporal_features.append(block(x))

        # Aggregate multi-scale features
        x = torch.stack(temporal_features, dim=1).mean(dim=1)

        # Self-attention with residual connections
        attention_weights = []
        for attn_layer in self.attention_layers:
            x, attn_w = attn_layer(x)
            attention_weights.append(attn_w)

        # Gated residual network
        x = self.grn(x)

        # Global pooling
        x_pooled = x.mean(dim=1)  # [batch, d_model]

        # Multi-task predictions
        glucose_pred = self.glucose_head(x_pooled)
        risk_pred = torch.sigmoid(self.risk_head(x_pooled))
        uncertainty = F.softplus(self.uncertainty_head(x_pooled))

        outputs = {
            'glucose': glucose_pred,
            'risk': risk_pred,
            'uncertainty': uncertainty
        }

        if return_attention:
            outputs['attention'] = attention_weights

        return outputs
