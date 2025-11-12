import torch
import torch.nn as nn

#First model 3 hidden layers with ReLU

class TestModel(nn.Module):
    
    def __init__(self, input_features, output_features=2):
        super().__init__()
        self.input_features = input_features
        self.model = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, output_features)
        )

    def forward(self, x):
        return self.model(x)