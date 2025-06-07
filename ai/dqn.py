import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
import os
from collections import deque, namedtuple

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.xavier_uniform_(self.weight_sigma)
        nn.init.zeros_(self.bias_mu)
        nn.init.zeros_(self.bias_sigma)

    def forward(self, x):
        # print(f"NoisyLinear input shape: {x.shape}")
        x = x.contiguous()  # 確保連續性
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected input shape (batch_size, {self.in_features}), got {x.shape}")
        # print(f"Before matmul, x shape: {x.shape}")
        eps_in = torch.randn(self.in_features).to(x.device)
        eps_out = torch.randn(self.out_features).to(x.device)
        # 修正權重計算，移除 unsqueeze(0)
        weight_noise = eps_out.unsqueeze(1) * eps_in  # 形狀 (512, 7744)
        weight = self.weight_mu + self.weight_sigma * weight_noise
        # print(f"Weight shape: {weight.shape}")
        bias = self.bias_mu + self.bias_sigma * eps_out
        return x @ weight.transpose(0, 1) + bias

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=3, stride=1, padding=1)  # 減至 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 減至 32
        def conv_output_shape(h, w, kernel_size=3, stride=1, padding=0):
            return ((h + 2 * padding - kernel_size) // stride) + 1
        h = conv_output_shape(state_dim[1], state_dim[2], stride=1, padding=1)
        w = conv_output_shape(state_dim[2], state_dim[2], stride=1, padding=1)
        h = conv_output_shape(h, w, stride=2, padding=1)
        w = conv_output_shape(w, w, stride=2, padding=1)
        conv_out_size = 32 * h * w  # 調整為 32 通道
        self.fc1 = NoisyLinear(conv_out_size, 128)  # 減至 128
        self.fc2_value = NoisyLinear(128, 1)
        self.fc2_advantage = NoisyLinear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)
        x = x.contiguous()
        x = F.relu(self.fc1(x))
        value = self.fc2_value(x)
        advantage = self.fc2_advantage(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        self.fc1.reset_parameters()
        self.fc2_value.reset_parameters()
        self.fc2_advantage.reset_parameters()