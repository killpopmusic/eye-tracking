import torch
import torch.nn as nn

class GazeRegressorMin(nn.Module):
    def __init__(self, input_features, output_dim: int = 2):
        super().__init__()
        self.input_features = input_features
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)
