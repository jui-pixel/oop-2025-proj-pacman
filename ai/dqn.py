"""
定義深度 Q 網路 (DQN) 模型，使用卷積神經網路 (CNN) 處理 Pac-Man 遊戲的狀態輸入。
模型預測每個動作的 Q 值，包含卷積層、批次正規化層和全連接層。
此模型設計與 agent.py 中的 DQNAgent 類兼容，支持優先級經驗回放和 Double DQN 訓練。
"""

import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化 DQN 模型，包含 3 層卷積層（帶批次正規化）和 2 層全連接層。
        模型結構經過優化，增加深度並添加正則化，適配 Pac-Man 遊戲的複雜狀態。

        Args:
            input_dim (Tuple[int, int, int]): 輸入維度 (高度, 寬度, 通道數)，例如 (31, 28, 6)。
                - 高度和寬度來自迷宮尺寸。
                - 通道數 6 分別表示 Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂和牆壁。
            output_dim (int): 輸出維度，對應動作數量（例如 4，表示上、下、左、右）。
        """
        super(DQN, self).__init__()
        self.input_dim = input_dim  # 保存輸入維度，用於計算卷積層輸出

        # 定義卷積層序列，包含 3 層卷積，每層後添加批次正規化和激活函數
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[2], 32, kernel_size=3, stride=1, padding=1),  # 第一層：輸入通道（6）-> 32
            # 使用 3x3 卷積核，padding=1 保持空間尺寸，stride=1 逐步提取特徵
            nn.BatchNorm2d(32),  # 批次正規化，穩定特徵分佈並加速訓練
            nn.ReLU(),  # ReLU 激活函數，增加非線性，提升特徵表達能力
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 第二層：32 -> 64
            nn.BatchNorm2d(64),  # 批次正規化
            nn.ReLU(),  # 激活
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 第三層：64 -> 128
            # stride=2 縮減空間尺寸，提升計算效率，減少參數量
            nn.BatchNorm2d(128),  # 批次正規化
            nn.ReLU()  # 激活
        )

        # 計算卷積層輸出大小，用於設計全連接層的輸入
        conv_out_size = self._get_conv_out(input_dim)

        # 定義全連接層序列，將卷積特徵映射到動作 Q 值
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),  # 將卷積展平輸出映射到 1024 維
            nn.ReLU(),  # 激活函數
            nn.Dropout(0.2),  # Dropout 層，降低過擬合風險，設置丟棄率為 0.2
            nn.Linear(1024, output_dim)  # 映射到動作數量，輸出每個動作的 Q 值
        )

    def _get_conv_out(self, shape):
        """
        計算卷積層輸出的展平大小，用於全連接層的輸入設計。
        通過模擬前向傳播計算卷積層輸出尺寸，確保全連接層設計正確。

        Args:
            shape (Tuple[int, int, int]): 輸入張量的形狀 (高度, 寬度, 通道數)。

        Returns:
            int: 展平後的輸出大小（元素總數），用於全連接層輸入。
        """
        # 使用零張量模擬前向傳播，計算卷積輸出尺寸
        # 輸入形狀為 (batch_size=1, channels, height, width)
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))
        # 計算展平後的元素數量：batch_size 外的所有維度相乘
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        前向傳播，處理輸入狀態並輸出每個動作的 Q 值。
        輸入經過卷積層提取特徵，再通過全連接層映射到動作空間。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。
                - 典型形狀例如 (batch_size, 6, 31, 28)。
                - 必須與初始化時的 input_dim 匹配。

        Returns:
            torch.Tensor: 每個動作的 Q 值，形狀為 (batch_size, output_dim)。
                - 例如 (batch_size, 4)，表示 4 個動作的 Q 值。
        """
        x = self.conv(x)  # 通過卷積層提取空間特徵
        x = x.reshape(x.size(0), -1)  # 展平為一維向量，形狀為 (batch_size, conv_out_size)
        x = self.fc(x)  # 通過全連接層計算 Q 值
        return x