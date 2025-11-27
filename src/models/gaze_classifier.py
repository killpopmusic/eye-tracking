import torch
import torch.nn as nn

class GazeClassifier(nn.Module):
    def __init__(self, input_features, num_classes=9):
        super().__init__()
        self.input_features = input_features
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_features,128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)
