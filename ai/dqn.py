# ai/dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=SIGMA):
        """
        Initialize Noisy Linear Layer.
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
        """
        Reset parameters for weights and biases.
        """
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_sigma, self.sigma / np.sqrt(self.in_features))
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_sigma, self.sigma / np.sqrt(self.in_features))

    def reset_noise(self):
        """
        Reset noise for epsilon values.
        """
        epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device) / np.sqrt(self.in_features)
        epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device) / np.sqrt(self.out_features)
        self.eps_in = torch.sign(epsilon_in) * torch.sqrt(torch.abs(epsilon_in))
        self.eps_out = torch.sign(epsilon_out) * torch.sqrt(torch.abs(epsilon_out))

    def forward(self, x):
        """
        Forward pass with noisy linear transformation.
        """
        x = x.contiguous()
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected input shape (batch_size, {self.in_features}), got {x.shape}")
        if not hasattr(self, 'eps_in'):
            self.reset_noise()
        weight_noise = self.eps_out.unsqueeze(1) * self.eps_in
        weight = self.weight_mu + self.weight_sigma * weight_noise
        bias = self.bias_mu + self.bias_sigma * self.eps_out
        weight_sigma_mean = self.weight_sigma.abs().mean().item()
        bias_sigma_mean = self.bias_sigma.abs().mean().item()
        return x @ weight.transpose(0, 1) + bias, {"weight_sigma_mean": weight_sigma_mean, "bias_sigma_mean": bias_sigma_mean}

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_conv_layers=3):
        """
        Initialize DQN network.
        """
        super(DQN, self).__init__()
        conv_layers = []
        in_channels = state_dim[0]
        channels = [16, 32, 64][:num_conv_layers]
        h, w = state_dim[1], state_dim[2]
        for i, out_channels in enumerate(channels):
            stride = 2 if i == num_conv_layers - 1 else 1
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            h = ((h + 2 * 1 - 3) // stride) + 1
            w = ((w + 2 * 1 - 3) // stride) + 1
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        conv_out_size = in_channels * h * w
        self.fc1 = NoisyLinear(conv_out_size, 256)
        self.fc2_value = NoisyLinear(256, 1)
        self.fc2_advantage = NoisyLinear(256, action_dim)

    def forward(self, x):
        """
        Forward pass to compute Q-values.
        """
        x = self.conv(x)
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)
        x, fc1_metrics = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        value, value_metrics = self.fc2_value(x)
        advantage, advantage_metrics = self.fc2_advantage(x)
        noise_metrics = {
            'fc1_weight_sigma_mean': fc1_metrics['weight_sigma_mean'],
            'fc1_bias_sigma_mean': fc1_metrics['bias_sigma_mean'],
            'value_weight_sigma_mean': value_metrics['weight_sigma_mean'],
            'value_bias_sigma_mean': value_metrics['bias_sigma_mean'],
            'advantage_weight_sigma_mean': advantage_metrics['weight_sigma_mean'],
            'advantage_bias_sigma_mean': advantage_metrics['bias_sigma_mean']
        }
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, noise_metrics

    def reset_noise(self):
        """
        Reset noise for all noisy layers.
        """
        self.fc1.reset_noise()
        self.fc2_value.reset_noise()
        self.fc2_advantage.reset_noise()