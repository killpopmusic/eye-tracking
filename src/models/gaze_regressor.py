import torch
import torch.nn as nn

class GazeRegressor(nn.Module):
    def __init__(self, input_features, output_dim: int = 2):
        super().__init__()
        self.input_features = input_features
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)
