# ai/dqn.py
"""
定義深度 Q 網路 (DQN) 模型，使用卷積神經網路處理 Pac-Man 遊戲的狀態輸入。
負責預測每個動作的 Q 值。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化 DQN 模型，包含卷積層和全連接層。

        Args:
            input_dim (Tuple[int, int, int]): 輸入維度 (高度, 寬度, 通道數)。
            output_dim (int): 輸出維度（動作數量）。
        """
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 額外卷積層
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _get_conv_out(self, shape):
        """
        計算卷積層輸出的大小。

        Args:
            shape (Tuple[int, int, int]): 輸入張量的形狀 (高度, 寬度, 通道數)。

        Returns:
            int: 展平後的輸出大小。
        """
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        前向傳播，處理輸入狀態並輸出 Q 值。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 每個動作的 Q 值，形狀為 (batch_size, output_dim)。
        """
        x = self.conv(x)  # 通過卷積層
        x = x.reshape(x.size(0), -1)  # 展平
        x = self.fc(x)  # 通過全連接層
        return x