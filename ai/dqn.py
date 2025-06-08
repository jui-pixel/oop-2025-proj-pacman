# ai/dqn.py
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
        """
        初始化嘈雜線性層，添加噪聲以增強探索。

        Args:
            in_features (int): 輸入特徵數。
            out_features (int): 輸出特徵數。
            sigma (float): 噪聲標準差，控制噪聲大小。
        """
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
        """初始化參數，使用 Xavier 初始化並將偏置設為 0。"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.xavier_uniform_(self.weight_sigma)
        nn.init.zeros_(self.bias_mu)
        nn.init.zeros_(self.bias_sigma)

    def reset_noise(self):
        """重新生成噪聲，保持參數不變。"""
        self.eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        self.eps_out = torch.randn(self.out_features, device=self.weight_mu.device)

    def forward(self, x):
        """
        前向傳播，加入噪聲計算輸出。

        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, in_features)。

        Returns:
            torch.Tensor: 輸出張量，形狀為 (batch_size, out_features)。
        """
        x = x.contiguous()
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected input shape (batch_size, {self.in_features}), got {x.shape}")
        if not hasattr(self, 'eps_in'):
            self.reset_noise()
        weight_noise = self.eps_out.unsqueeze(1) * self.eps_in
        weight = self.weight_mu + self.weight_sigma * weight_noise
        bias = self.bias_mu + self.bias_sigma * self.eps_out
        return x @ weight.transpose(0, 1) + bias

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        初始化 DQN 網絡，使用卷積層和嘈雜線性層。

        Args:
            state_dim (tuple): 狀態空間維度，例如 (6, 21, 21)。
            action_dim (int): 動作空間維度，例如 4。
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        def conv_output_shape(h, w, kernel_size=3, stride=1, padding=0):
            return ((h + 2 * padding - kernel_size) // stride) + 1
        h = conv_output_shape(state_dim[1], state_dim[2], stride=1, padding=1)
        w = conv_output_shape(state_dim[2], state_dim[2], stride=1, padding=1)
        h = conv_output_shape(h, w, stride=2, padding=1)
        w = conv_output_shape(w, w, stride=2, padding=1)
        conv_out_size = 32 * h * w
        self.fc1 = NoisyLinear(conv_out_size, 128)
        self.fc2_value = NoisyLinear(128, 1)
        self.fc2_advantage = NoisyLinear(128, action_dim)

    def forward(self, x):
        """
        前向傳播，計算每個動作的 Q 值。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, 6, 21, 21)。

        Returns:
            torch.Tensor: Q 值張量，形狀為 (batch_size, action_dim)。
        """
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
        """重置嘈雜層的噪聲。"""
        self.fc1.reset_noise()
        self.fc2_value.reset_noise()
        self.fc2_advantage.reset_noise()