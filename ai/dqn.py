# ai/dqn.py
"""
定義深度 Q 網路 (DQN) 模型，使用卷積神經網路 (CNN) 處理 Pac-Man 遊戲的狀態輸入。
模型預測每個動作的 Q 值，包含卷積層、批次正規化層和全連接層。
"""

import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化 DQN 模型，包含 4 層卷積層（帶批次正規化）和 3 層全連接層。

        Args:
            input_dim (Tuple[int, int, int]): 輸入維度 (高度, 寬度, 通道數)，例如 (31, 28, 6)。
            output_dim (int): 輸出維度，對應動作數量（例如 4，表示上、下、左、右）。
        """
        super(DQN, self).__init__()
        self.input_dim = input_dim  # 保存輸入維度以便後續計算

        # 定義卷積層序列，包含 4 層卷積，每層後（除最後一層外）添加批次正規化
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[2], 32, kernel_size=3, stride=1, padding=1),  # 第一層：輸入通道 -> 32
            nn.BatchNorm2d(32),  # 批次正規化，穩定特徵分佈
            nn.ReLU(),  # 激活函數，增加非線性
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 第二層：32 -> 64
            nn.BatchNorm2d(64),  # 批次正規化
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 第三層：64 -> 128
            nn.BatchNorm2d(128),  # 批次正規化
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 第四層：128 -> 128
            nn.ReLU()  # 最後一層無批次正規化，直接激活
        )

        # 計算卷積層輸出大小，用於全連接層的輸入
        conv_out_size = self._get_conv_out(input_dim)

        # 定義全連接層序列，映射卷積輸出到動作 Q 值
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),  # 將卷積展平輸出映射到 1024 維
            nn.ReLU(),  # 激活函數
            nn.Dropout(0.5),  # Dropout 防止過擬合
            nn.Linear(1024, 512),  # 1024 -> 512
            nn.ReLU(),
            nn.Linear(512, output_dim)  # 512 -> 動作數（Q 值）
        )

    def _get_conv_out(self, shape):
        """
        計算卷積層輸出的展平大小，用於全連接層的輸入設計。

        Args:
            shape (Tuple[int, int, int]): 輸入張量的形狀 (高度, 寬度, 通道數)。

        Returns:
            int: 展平後的輸出大小（元素總數）。
        """
        # 使用零張量模擬前向傳播，計算卷積輸出尺寸
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        前向傳播，處理輸入狀態並輸出每個動作的 Q 值。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 每個動作的 Q 值，形狀為 (batch_size, output_dim)。
        """
        x = self.conv(x)  # 通過卷積層提取特徵
        x = x.reshape(x.size(0), -1)  # 展平為一維向量
        x = self.fc(x)  # 通過全連接層計算 Q 值
        return x