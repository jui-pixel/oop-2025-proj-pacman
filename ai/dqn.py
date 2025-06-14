# ai/dqn.py
# 導入 PyTorch 相關模組，用於建立神經網絡
import torch
# 導入 PyTorch 的神經網絡模組，提供層類型（如線性層、卷積層）
import torch.nn as nn
# 導入 PyTorch 的函數操作模組，例如激活函數
import torch.nn.functional as F
# 導入 numpy，用於數學計算和陣列操作
import numpy as np
# 從 config 檔案匯入常數參數，例如 SIGMA（用於噪聲層）
from config import *

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=SIGMA):
        """
        初始化 Noisy Linear 層，這是一個帶有可學習噪聲的線性層。

        Noisy Linear 層在傳統線性層的基礎上添加了隨機噪聲，用於強化學習中的探索策略。

        Args:
            in_features (int): 輸入特徵數（輸入神經元數量）。
            out_features (int): 輸出特徵數（輸出神經元數量）。
            sigma (float): 噪聲的初始標準差，預設從 config 檔案的 SIGMA 取得。
        """
        # 呼叫父類 nn.Module 的初始化函數
        super(NoisyLinear, self).__init__()
        # 儲存輸入和輸出特徵數
        self.in_features = in_features
        self.out_features = out_features
        # 儲存噪聲標準差
        self.sigma = sigma
        # 定義可學習的權重均值參數，形狀為 (out_features, in_features)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        # 定義可學習的權重噪聲標準差參數，形狀同上
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        # 定義可學習的偏置均值參數，形狀為 (out_features,)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        # 定義可學習的偏置噪聲標準差參數，形狀同上
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        # 初始化參數
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置權重和偏置的參數。

        這個方法用來初始化層的參數，確保網絡從合理的初始值開始訓練。
        """
        # 使用 Xavier 均勻分佈初始化權重均值，適合激活函數如 ReLU
        nn.init.xavier_uniform_(self.weight_mu)
        # 初始化權重噪聲標準差為 sigma / sqrt(in_features)，確保噪聲適當縮放
        nn.init.constant_(self.weight_sigma, self.sigma / np.sqrt(self.in_features))
        # 初始化偏置均值為 0
        nn.init.zeros_(self.bias_mu)
        # 初始化偏置噪聲標準差為 sigma / sqrt(in_features)
        nn.init.constant_(self.bias_sigma, self.sigma / np.sqrt(self.in_features))

    def reset_noise(self):
        """
        重置噪聲值，用於每次前向傳播時生成新的隨機噪聲。

        噪聲採用因子分解方式生成，提升計算效率。
        """
        # 生成輸入維度的隨機噪聲，形狀為 (in_features,)
        epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device) / np.sqrt(self.in_features)
        # 生成輸出維度的隨機噪聲，形狀為 (out_features,)
        epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device) / np.sqrt(self.out_features)
        # 對噪聲進行簽名和平方根處理，生成因子化噪聲
        self.eps_in = torch.sign(epsilon_in) * torch.sqrt(torch.abs(epsilon_in))
        self.eps_out = torch.sign(epsilon_out) * torch.sqrt(torch.abs(epsilon_out))

    def forward(self, x):
        """
        執行帶噪聲的線性層前向傳播。

        將輸入 x 進行線性變換，同時加入隨機噪聲以增強探索。

        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, in_features)。

        Returns:
            tuple: (輸出張量, 噪聲指標字典)。
        """
        # 確保輸入張量是連續的，提升計算效率
        x = x.contiguous()
        # 檢查輸入張量的形狀是否正確
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected input shape (batch_size, {self.in_features}), got {x.shape}")
        # 如果尚未生成噪聲，則初始化
        if not hasattr(self, 'eps_in'):
            self.reset_noise()
        # 計算權重噪聲，形狀為 (out_features, in_features)
        weight_noise = self.eps_out.unsqueeze(1) * self.eps_in
        # 結合均值和噪聲生成最終權重
        weight = self.weight_mu + self.weight_sigma * weight_noise
        # 結合均值和噪聲生成最終偏置
        bias = self.bias_mu + self.bias_sigma * self.eps_out
        # 計算權重和偏置噪聲標準差的平均值，用於監控
        weight_sigma_mean = self.weight_sigma.abs().mean().item()
        bias_sigma_mean = self.bias_sigma.abs().mean().item()
        # 執行線性變換：y = x @ weight^T + bias
        return x @ weight.transpose(0, 1) + bias, {"weight_sigma_mean": weight_sigma_mean, "bias_sigma_mean": bias_sigma_mean}

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_conv_layers=3, sigma=SIGMA):
        """
        初始化 DQN（Deep Q-Network）網絡。

        DQN 是一個用於強化學習的神經網絡，結合卷積層和全連接層，輸出動作的 Q 值。

        Args:
            state_dim (tuple): 狀態維度，形狀為 (通道數, 高度, 寬度)。
            action_dim (int): 動作數量（輸出 Q 值的數量）。
            num_conv_layers (int): 卷積層數量，預設為 3。
            sigma (float): NoisyLinear 層的噪聲標準差。
        """
        # 呼叫父類 nn.Module 的初始化函數
        super(DQN, self).__init__()
        # 建立卷積層序列
        conv_layers = []
        # 輸入通道數，從狀態維度的第一維取得
        in_channels = state_dim[0]
        # 定義每層卷積的輸出通道數，最多取前三個
        channels = [16, 32, 64][:num_conv_layers]
        # 取得輸入狀態的高度和寬度
        h, w = state_dim[1], state_dim[2]
        # 逐層添加卷積層、批次正規化和 ReLU 激活函數
        for i, out_channels in enumerate(channels):
            # 最後一層卷積使用 stride=2，其餘使用 stride=1
            stride = 2 if i == num_conv_layers - 1 else 1
            conv_layers.extend([
                # 添加 2D 卷積層，內核大小為 3，填充為 1
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                # 添加批次正規化層，穩定訓練
                nn.BatchNorm2d(out_channels),
                # 添加 ReLU 激活函數，增加非線性
                nn.ReLU(),
            ])
            # 計算卷積後的輸出高度和寬度
            h = ((h + 2 * 1 - 3) // stride) + 1
            w = ((w + 2 * 1 - 3) // stride) + 1
            # 更新輸入通道數為當前層的輸出通道數
            in_channels = out_channels
        # 將卷積層序列封裝為 Sequential 容器
        self.conv = nn.Sequential(*conv_layers)
        # 計算卷積層輸出展平後的大小
        conv_out_size = in_channels * h * w
        # 添加第一個全連接層（NoisyLinear），從卷積輸出到 256 維
        self.fc1 = NoisyLinear(conv_out_size, 256, sigma)
        # 添加價值流的全連接層，輸出 1 維（狀態價值）
        self.fc2_value = NoisyLinear(256, 1, sigma)
        # 添加優勢流的全連接層，輸出 action_dim 維（動作優勢）
        self.fc2_advantage = NoisyLinear(256, action_dim, sigma)

    def forward(self, x):
        """
        執行 DQN 網絡的前向傳播，計算 Q 值。

        使用 Dueling DQN 架構，將 Q 值分解為價值（value）和優勢（advantage）。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。

        Returns:
            tuple: (Q 值張量, 噪聲指標字典)。
        """
        # 通過卷積層處理輸入
        x = self.conv(x)
        # 獲取批次大小
        batch_size = x.size(0)
        # 將卷積輸出展平為 (batch_size, -1)
        x = x.contiguous().view(batch_size, -1)
        # 通過第一個全連接層，並獲取噪聲指標
        x, fc1_metrics = self.fc1(x)
        # 應用 ReLU 激活函數
        x = F.relu(x)
        # 在訓練模式下應用 dropout（丟棄率 0.3），防止過擬合
        x = F.dropout(x, p=0.3, training=self.training)
        # 通過價值流全連接層，計算狀態價值
        value, value_metrics = self.fc2_value(x)
        # 通過優勢流全連接層，計算動作優勢
        advantage, advantage_metrics = self.fc2_advantage(x)
        # 收集所有噪聲指標
        noise_metrics = {
            'fc1_weight_sigma_mean': fc1_metrics['weight_sigma_mean'],
            'fc1_bias_sigma_mean': fc1_metrics['bias_sigma_mean'],
            'value_weight_sigma_mean': value_metrics['weight_sigma_mean'],
            'value_bias_sigma_mean': value_metrics['bias_sigma_mean'],
            'advantage_weight_sigma_mean': advantage_metrics['weight_sigma_mean'],
            'advantage_bias_sigma_mean': advantage_metrics['bias_sigma_mean']
        }
        # 計算最終 Q 值：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        # 返回 Q 值和噪聲指標
        return q_values, noise_metrics

    def reset_noise(self):
        """
        重置所有 NoisyLinear 層的噪聲。

        在每次訓練迭代時調用，確保生成新的隨機噪聲。
        """
        # 重置第一個全連接層的噪聲
        self.fc1.reset_noise()
        # 重置價值流全連接層的噪聲
        self.fc2_value.reset_noise()
        # 重置優勢流全連接層的噪聲
        self.fc2_advantage.reset_noise()