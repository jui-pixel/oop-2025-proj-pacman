"""
定義基礎深度 Q 網路 (DQN) 模型，使用卷積神經網路 (CNN) 處理 Pac-Man 遊戲的狀態輸入。
此模型設計與 agent.py 中的 DQNAgent 類兼容，支援基礎 DQN 的訓練。
"""

import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化基礎 DQN 模型，包含 2 層卷積層和 2 層全連接層。

        Args:
            input_dim (Tuple[int, int, int]): 輸入維度 (通道數, 高度, 寬度)，例如 (6, 31, 28)。
                - 通道數 6 分別表示 Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂和牆壁。
            output_dim (int): 輸出維度，對應動作數量（例如 4，表示上、下、左、右）。
        """
        super(DQN, self).__init__()
        self.input_channels = input_dim[0]
        self.input_height = input_dim[1]
        self.input_width = input_dim[2]

        # 定義卷積層序列，提取特徵
        self.feature = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1),  # (6, H, W) -> (32, H, W)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, H, W) -> (64, H/2, W/2)
            nn.ReLU()
        )

        # 計算卷積層輸出的展平大小
        self.conv_out_size = self._get_conv_out(input_dim)

        # 定義全連接層
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),  # 展平後映射到 512 維
            nn.ReLU(),
            nn.Linear(512, output_dim)  # 輸出動作 Q 值
        )

    def _get_conv_out(self, input_shape):
        """
        計算卷積層輸出的展平大小。

        Args:
            input_shape (Tuple[int, int, int]): 輸入張量的形狀 (通道數, 高度, 寬度)。
        Returns:
            int: 展平後的輸出大小。
        """
        o = self.feature(torch.zeros(1, input_shape[0], input_shape[1], input_shape[2]))
        return int(np.prod(o.size()))

    def forward(self, state):
        """
        前向傳播，計算動作的 Q 值。

        Args:
            state (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 輸出 Q 值張量，形狀為 (batch_size, output_dim)。
        """
        features = self.feature(state)
        features = features.view(features.size(0), -1)
        q_values = self.fc(features)
        return q_values