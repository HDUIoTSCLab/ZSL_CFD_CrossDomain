import torch.nn as nn
import torch

class TransformerSynthesizer(nn.Module):
    def __init__(self, base_dim, max_parts=3, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.base_dim = base_dim
        self.max_parts = max_parts
        self.pos_embed = nn.Parameter(torch.randn(max_parts, base_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_dim, nhead=nhead,
            dim_feedforward=base_dim * 2, dropout=dropout,
            activation='relu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(base_dim, base_dim)

    def forward(self, parts):
        if len(parts) < self.max_parts:
            pad = [torch.zeros_like(parts[0]) for _ in range(self.max_parts - len(parts))]
            parts += pad
        x = torch.cat(parts, dim=0).unsqueeze(0)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.output_proj(x)