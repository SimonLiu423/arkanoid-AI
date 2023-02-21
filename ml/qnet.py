import numpy as np
import torch.nn as nn


class QNet(nn.Module):    # (814, )
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)
