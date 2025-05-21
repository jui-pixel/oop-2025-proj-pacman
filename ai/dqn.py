# ai/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim  # (height, width, channels)
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[2], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x