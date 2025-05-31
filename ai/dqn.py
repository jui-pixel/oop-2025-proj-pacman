# ai/dqn.py
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
            input_dim (Tuple[int, int, int]): 輸入維度 (通道數, 高度, 寬度)，例如 (6, 31, 28)。
                - 高度和寬度來自迷宮尺寸。
                - 通道數 6 分別表示 Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂和牆壁。
            output_dim (int): 輸出維度，對應動作數量（例如 4，表示上、下、左、右）。
        """
        super(DuelingDQN, self).__init__()
        # input_dim 現在應該是 (C, H, W)
        self.input_channels = input_dim[0]
        self.input_height = input_dim[1]
        self.input_width = input_dim[2]

        # 定義卷積層序列，提取特徵
        # PyTorch 的 Conv2d 輸入是 (Batch_Size, Channels, Height, Width)
        # 這裡的 in_channels 應該是 input_dim[0] (通道數)
        self.feature = nn.Sequential(
            # 第一層: (C, H, W) -> (32, H', W')
            # 選擇較小的 kernel_size 和 stride，以確保不會超出輸入尺寸
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1), # 保持尺寸不變或略微縮小
            nn.BatchNorm2d(32), # 批次正規化
            nn.ReLU(), # 激活函數

            # 第二層: (32, H', W') -> (64, H'', W'')
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 尺寸減半
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 第三層: (64, H'', W'') -> (64, H''', W''')
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 保持尺寸不變或略微縮小
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 計算卷積層輸出的展平大小，用於後續全連接層的輸入
        # 這裡傳入 _get_conv_out 的 shape 應該是原始的 (C, H, W) 格式
        self.conv_out_size = self._get_conv_out(input_dim) 

        # 定義價值流（Value Stream），估計狀態的價值
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),  # 展平後映射到 256 維
            nn.ReLU(),  # 激活函數
            nn.Dropout(0.1),  # Dropout 降低過擬合風險
            nn.Linear(256, 1)  # 輸出單一價值 V(s)
        )

        # 定義優勢流（Advantage Stream），估計每個動作的優勢
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),  # 展平後映射到 256 維
            nn.ReLU(),  # 激活函數
            nn.Dropout(0.1),  # Dropout 降低過擬合風險
            nn.Linear(256, output_dim)  # 輸出每個動作的優勢 A(s, a)
        )

    def _get_conv_out(self, input_shape):
        """
        計算卷積層輸出的展平大小，用於後續層的輸入設計。

        Args:
            input_shape (Tuple[int, int, int]): 輸入張量的形狀 (通道數, 高度, 寬度)。
        Returns:
            int: 展平後的輸出大小（元素總數）。
        """
        # 創建一個假輸入，形狀為 (Batch_Size, Channels, Height, Width)
        # input_shape[0] 是 Channels, input_shape[1] 是 Height, input_shape[2] 是 Width
        o = self.feature(torch.zeros(1, input_shape[0], input_shape[1], input_shape[2]))
        return int(np.prod(o.size()))

    def forward(self, state):
        """
        前向傳播，計算 Dueling Q 值。

        Args:
            state (torch.Tensor): 輸入狀態張量，形狀應為 (Batch_Size, Channels, Height, Width)。

        Returns:
            torch.Tensor: 輸出 Q 值張量。
        """
        # 通過卷積層提取特徵
        features = self.feature(state)
        # 展平特徵
        features = features.view(features.size(0), -1) # 自動推斷展平大小

        # 分別通過價值流和優勢流
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 合併價值和優勢以計算 Q 值
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values