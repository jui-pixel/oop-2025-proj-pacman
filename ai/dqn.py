# ai/dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pickle
import os
from collections import deque, namedtuple

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        """
        初始化嘈雜線性層（Noisy Linear Layer），用於在深度強化學習中引入隨機噪聲以增強探索。

        原理：
        - 傳統的 DQN 使用 ε-貪婪策略進行探索，但可能導致探索不充分或過於隨機。
        - NoisyLinear 層通過在權重和偏置中添加可學習的噪聲，動態調整探索行為。
        - 噪聲公式：w = μ_w + σ_w * ε_w，b = μ_b + σ_b * ε_b，其中：
          - μ_w, μ_b：權重和偏置的均值（可學習參數）。
          - σ_w, σ_b：權重和偏置的標準差（可學習參數）。
          - ε_w, ε_b：隨機噪聲，通常為標準正態分佈。
        - 這種方法使網絡在訓練過程中自動學習探索與利用的平衡，無需手動調整 ε。

        Args:
            in_features (int): 輸入特徵數（輸入層的神經元數）。
            out_features (int): 輸出特徵數（輸出層的神經元數）。
            sigma (float): 噪聲標準差初始值，控制噪聲強度。
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features  # 輸入特徵數
        self.out_features = out_features  # 輸出特徵數
        self.sigma = sigma  # 噪聲標準差
        # 定義可學習參數：權重均值、權重標準差、偏置均值、偏置標準差
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()  # 初始化參數

    def reset_parameters(self):
        """
        初始化參數，使用 Xavier 初始化和零偏置。

        原理：
        - Xavier 初始化（也稱 Glorot 初始化）用於權重初始化，確保輸入和輸出的方差一致，促進穩定訓練。
        - 公式：w ~ Uniform(-√(6/(in_features + out_features)), √(6/(in_features + out_features)))
        - 偏置初始化為零，噪聲標準差也初始化為零，確保初始行為接近標準線性層。
        """
        nn.init.xavier_uniform_(self.weight_mu)  # 使用 Xavier 初始化權重均值
        nn.init.xavier_uniform_(self.weight_sigma)  # 使用 Xavier 初始化權重標準差
        nn.init.zeros_(self.bias_mu)  # 偏置均值初始化為零
        nn.init.zeros_(self.bias_sigma)  # 偏置標準差初始化為零

    def reset_noise(self):
        """
        重新生成噪聲，保持參數不變。

        原理：
        - 每次前向傳播時，使用新的隨機噪聲 ε，確保探索行為的隨機性。
        - 噪聲從標準正態分佈 N(0, 1) 中抽樣，獨立於輸入和輸出維度。
        """
        self.eps_in = torch.randn(self.in_features, device=self.weight_mu.device)  # 輸入維度噪聲
        self.eps_out = torch.randn(self.out_features, device=self.weight_mu.device)  # 輸出維度噪聲

    def forward(self, x):
        """
        前向傳播，計算帶噪聲的線性變換。

        原理：
        - 計算公式：y = (μ_w + σ_w * ε_w) * x + (μ_b + σ_b * ε_b)
        - 噪聲 ε_w 和 ε_b 在每次前向傳播時重新生成，確保動態探索。
        - 噪聲矩陣 ε_w 由外積生成：ε_w = eps_out.unsqueeze(1) * eps_in，形狀為 (out_features, in_features)。

        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, in_features)。

        Returns:
            torch.Tensor: 輸出張量，形狀為 (batch_size, out_features)。
        """
        x = x.contiguous()  # 確保輸入張量內存連續
        # 驗證輸入形狀
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"預期輸入形狀 (batch_size, {self.in_features})，得到 {x.shape}")
        if not hasattr(self, 'eps_in'):
            self.reset_noise()  # 若未初始化噪聲，則生成
        # 計算帶噪聲的權重
        weight_noise = self.eps_out.unsqueeze(1) * self.eps_in  # 噪聲矩陣
        weight = self.weight_mu + self.weight_sigma * weight_noise  # 最終權重
        # 計算帶噪聲的偏置
        bias = self.bias_mu + self.bias_sigma * self.eps_out  # 最終偏置
        # 計算線性變換：y = x * w^T + b
        return x @ weight.transpose(0, 1) + bias

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        初始化 DQN 網絡，使用卷積層和嘈雜線性層實現 Q 值函數逼近。

        原理：
        - DQN（Deep Q-Network）是一種深度強化學習模型，通過神經網絡逼近 Q 值函數 Q(s, a)。
        - 本實現採用 Dueling DQN 結構，將 Q 值分解為狀態值 V(s) 和優勢函數 A(s, a)：
          - Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
          - V(s) 表示狀態的價值，A(s, a) 表示動作相對於平均動作的優勢。
        - 使用卷積層處理高維狀態（如圖像或格子數據），NoisyLinear 層增強探索。
        - 網絡結構：
          - 輸入：狀態 (batch_size, channels, height, width)
          - 卷積層：提取空間特徵
          - 全連接層：計算 V(s) 和 A(s, a)
          - 輸出：Q 值 (batch_size, action_dim)

        Args:
            state_dim (tuple): 狀態空間維度，例如 (6, 21, 21)，表示 6 通道的 21x21 格子。
            action_dim (int): 動作空間維度，例如 4（上、下、左、右）。
        """
        super(DQN, self).__init__()
        # 卷積層：提取空間特徵
        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=3, stride=1, padding=1)  # 第一卷積層
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 第二卷積層
        # 計算卷積層輸出尺寸
        def conv_output_shape(h, w, kernel_size=3, stride=1, padding=0):
            """
            計算卷積層輸出尺寸。

            公式：output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1
            """
            return ((h + 2 * padding - kernel_size) // stride) + 1
        # 計算第一卷積層輸出尺寸
        h = conv_output_shape(state_dim[1], state_dim[2], stride=1, padding=1)
        w = conv_output_shape(state_dim[2], state_dim[2], stride=1, padding=1)
        # 計算第二卷積層輸出尺寸
        h = conv_output_shape(h, w, stride=2, padding=1)
        w = conv_output_shape(w, w, stride=2, padding=1)
        conv_out_size = 32 * h * w  # 卷積層輸出總尺寸
        # 全連接層：Dueling DQN 的價值流和優勢流
        self.fc1 = NoisyLinear(conv_out_size, 128)  # 中間層
        self.fc2_value = NoisyLinear(128, 1)  # 價值流，輸出 V(s)
        self.fc2_advantage = NoisyLinear(128, action_dim)  # 優勢流，輸出 A(s, a)

    def forward(self, x):
        """
        前向傳播，計算每個動作的 Q 值。

        原理：
        - 輸入狀態經過卷積層提取特徵，展平後輸入全連接層。
        - 使用 Dueling DQN 結構計算 Q 值：
          - V(s)：狀態價值，由 fc2_value 輸出。
          - A(s, a)：動作優勢，由 fc2_advantage 輸出。
          - Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))，通過平均優勢校正確保穩定性。
        - ReLU 激活函數用於增加非線性，提升模型表達能力。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: Q 值張量，形狀為 (batch_size, action_dim)。
        """
        x = F.relu(self.conv1(x))  # 第一卷積層 + ReLU
        x = F.relu(self.conv2(x))  # 第二卷積層 + ReLU
        batch_size = x.size(0)  # 獲取批量大小
        x = x.contiguous().view(batch_size, -1)  # 展平卷積輸出
        x = x.contiguous()  # 確保內存連續
        x = F.relu(self.fc1(x))  # 中間層 + ReLU
        value = self.fc2_value(x)  # 計算狀態價值 V(s)
        advantage = self.fc2_advantage(x)  # 計算動作優勢 A(s, a)
        # 計算 Q 值：Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        """
        重置所有嘈雜層的噪聲。

        原理：
        - 在每次動作選擇或訓練前，重新生成噪聲以確保探索的隨機性。
        - 調用 NoisyLinear 的 reset_noise 方法，重置 fc1、fc2_value 和 fc2_advantage 的噪聲。
        """
        self.fc1.reset_noise()
        self.fc2_value.reset_noise()
        self.fc2_advantage.reset_noise()