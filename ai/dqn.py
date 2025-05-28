"""
定義深度 Q 網路 (Dueling DQN) 模型，使用卷積神經網路 (CNN) 處理 Pac-Man 遊戲的狀態輸入。
模型採用 Dueling 架構，分離價值流和優勢流，提升動作選擇效率。
此模型設計與 agent.py 中的 DQNAgent 類兼容，支持優先級經驗回放、Double DQN 和 n-step 學習。
"""

import torch
import torch.nn as nn
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化 Dueling DQN 模型，包含 3 層卷積層（帶批次正規化）、價值流和優勢流。
        模型結構經過優化，適配 Pac-Man 遊戲的複雜狀態。

        Args:
            input_dim (Tuple[int, int, int]): 輸入維度 (高度, 寬度, 通道數)，例如 (31, 28, 6)。
                - 高度和寬度來自迷宮尺寸。
                - 通道數 6 分別表示 Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂和牆壁。
            output_dim (int): 輸出維度，對應動作數量（例如 4，表示上、下、左、右）。
        """
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim  # 保存輸入維度，用於計算卷積層輸出

        # 定義卷積層序列，提取特徵，保留原有的批次正規化
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim[2], 16, kernel_size=3, stride=1, padding=1),  # 第一層：輸入通道（6）-> 16
            nn.BatchNorm2d(16),  # 批次正規化，穩定特徵分佈
            nn.ReLU(),  # ReLU 激活函數
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 第二層：16 -> 32
            nn.BatchNorm2d(32),  # 批次正規化
            nn.ReLU(),  # 激活
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 第三層：32 -> 64
            nn.BatchNorm2d(64),  # 批次正規化
            nn.ReLU()  # 激活
        )

        # 計算卷積層輸出大小，用於設計價值流和優勢流
        conv_out_size = self._get_conv_out(input_dim)

        # 定義價值流（Value Stream），估計狀態的總體價值
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),  # 展平後映射到 256 維
            nn.ReLU(),  # 激活函數
            nn.Dropout(0.1),  # Dropout 降低過擬合風險
            nn.Linear(256, 1)  # 輸出單一價值 V(s)
        )

        # 定義優勢流（Advantage Stream），估計每個動作的優勢
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),  # 展平後映射到 256 維
            nn.ReLU(),  # 激活函數
            nn.Dropout(0.1),  # Dropout 降低過擬合風險
            nn.Linear(256, output_dim)  # 輸出每個動作的優勢 A(s, a)
        )

    def _get_conv_out(self, shape):
        """
        計算卷積層輸出的展平大小，用於後續層的輸入設計。

        Args:
            shape (Tuple[int, int, int]): 輸入張量的形狀 (高度, 寬度, 通道數)。

        Returns:
            int: 展平後的輸出大小（元素總數）。
        """
        o = self.feature(torch.zeros(1, shape[2], shape[0], shape[1]))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        前向傳播，提取特徵並分離為價值流和優勢流，計算最終 Q 值。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 每個動作的 Q 值，形狀為 (batch_size, output_dim)。
        """
        x = self.feature(x)  # 通過卷積層提取特徵
        x = x.reshape(x.size(0), -1)  # 展平為一維向量
        value = self.value_stream(x)  # 計算狀態價值 V(s)
        advantage = self.advantage_stream(x)  # 計算動作優勢 A(s, a)
        # 結合價值和優勢，計算 Q 值：Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q