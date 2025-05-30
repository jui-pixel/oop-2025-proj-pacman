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
            input_dim (Tuple[int, int, int]): 輸入維度 (高度, 寬度, 通道數)，例如 (31, 28, 6)。
                - 高度和寬度來自迷宮尺寸。
                - 通道數 6 分別表示 Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂和牆壁。
            output_dim (int): 輸出維度，對應動作數量（例如 4，表示上、下、左、右）。
        """
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim  # 保存輸入維度 (H, W, C)

        # 定義卷積層序列，提取特徵
        # 注意：PyTorch 的 Conv2d 輸入是 (Batch_Size, Channels, Height, Width)
        # 所以在定義卷積層時，in_channels 應該是 input_dim 的第三個元素 (通道數)
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4), # input_dim[0] 是通道數
            nn.BatchNorm2d(32), # 批次正規化
            nn.ReLU(), # 激活函數
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 計算卷積層輸出的展平大小，用於後續全連接層的輸入
        # 這裡需要傳入的 input_dim 應是 (C, H, W) 才能讓 _get_conv_out 正確計算
        # 由於環境輸出的是 (C, H, W)，所以這裡的 input_dim 需要是 (C, H, W) 才能匹配
        # 如果你的環境輸出是 (H, W, C)，需要調整 _get_conv_out 的輸入
        self.conv_out_size = self._get_conv_out((input_dim[1], input_dim[2], input_dim[0])) # 這裡調整為 (H, W, C) 傳給 _get_conv_out

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

    def _get_conv_out(self, shape):
        """
        計算卷積層輸出的展平大小，用於後續層的輸入設計。

        Args:
            shape (Tuple[int, int, int]): 輸入張量的形狀 (高度, 寬度, 通道數)。
                                          這裡的 shape 應該是 (H, W, C)
        Returns:
            int: 展平後的輸出大小（元素總數）。
        """
        # 創建一個假輸入，注意其形狀應為 (Batch_Size, Channels, Height, Width)
        # 所以這裡的 shape[2] 是通道數，shape[0] 是高度，shape[1] 是寬度
        o = self.feature(torch.zeros(1, shape[2], shape[0], shape[1]))
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