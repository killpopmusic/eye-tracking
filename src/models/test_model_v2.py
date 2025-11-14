import torch
import torch.nn as nn

# Alternative MLP with LayerNorm and SiLU activations

class TestModelV2(nn.Module):
    
    def __init__(self, input_features, output_features=2):
        super().__init__()
        self.input_features = input_features
        self.model = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            
            nn.Linear(256, output_features)
        )

    def forward(self, x):
        return self.model(x)
