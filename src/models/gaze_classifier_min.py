import torch
import torch.nn as nn
#suited for eye+iris inputs only
class GazeClassifierMin(nn.Module):
    def __init__(self, input_features, num_classes=9):
        super().__init__()
        self.input_features = input_features
        self.num_classes = num_classes

        self.model = nn.Sequential(

            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.SiLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)
