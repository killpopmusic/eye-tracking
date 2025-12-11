import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)

class GazeResMLP(nn.Module):

    def __init__(self, input_features, num_classes=5, hidden_dim=128, num_blocks=4, dropout_rate=0.3):
        super().__init__()
        self.input_features = input_features
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.head(x)
