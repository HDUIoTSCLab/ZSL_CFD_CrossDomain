import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    def forward(self, x):
        return self.fc(x)