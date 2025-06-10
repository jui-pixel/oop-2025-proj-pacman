import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class NoisyLinear(nn.Module):
    """嘈雜線性層（Noisy Linear Layer），用於在深度強化學習中引入隨機噪聲以增強探索。

    原理：
    - 傳統 DQN 使用 ε-greedy 策略進行探索，可能導致探索不足或過於隨機。
    - NoisyLinear 透過在權重和偏置中添加可學習的噪聲，動態調整探索行為，無需手動調整 ε。
    - 數學公式：
      - 權重：w = μ_w + σ_w * ε_w
      - 偏置：b = μ_b + σ_b * ε_b
      - 其中 μ_w, μ_b 為可學習均值，σ_w, σ_b 為可學習標準差，ε_w, ε_b 為隨機噪聲（標準正態分佈）。
    - 在 Pac-Man 遊戲中，NoisyLinear 用於 DQN 的全連接層，幫助代理探索迷宮中的不同路徑。
    """
    def __init__(self, in_features, out_features, sigma=0.5):
        """初始化嘈雜線性層。

        Args:
            in_features (int): 輸入特徵數（輸入層神經元數，例如卷積層展平後的尺寸）。
            out_features (int): 輸出特徵數（輸出層神經元數，例如動作數或中間層尺寸）。
            sigma (float): 噪聲標準差初始值，控制噪聲強度（預設 0.5）。
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features  # 儲存輸入特徵數
        self.out_features = out_features  # 儲存輸出特徵數
        self.sigma = sigma  # 儲存噪聲標準差
        # 定義可學習參數
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))  # 權重均值
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))  # 權重標準差
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))  # 偏置均值
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))  # 偏置標準差
        self.reset_parameters()  # 初始化參數

    def reset_parameters(self):
        """初始化權重和偏置參數。

        原理：
        - 使用 Xavier 初始化（Glorot 初始化）設置權重均值和標準差，確保輸入和輸出方差一致，穩定訓練。
        - 公式：w ~ Uniform(-√(6/(in_features + out_features)), √(6/(in_features + out_features)))
        - 偏置均值和標準差初始化為零，確保初始行為接近標準線性層。
        """
        nn.init.xavier_uniform_(self.weight_mu)  # Xavier 初始化權重均值
        nn.init.constant_(self.weight_sigma, self.sigma / np.sqrt(self.in_features))  # Xavier 初始化權重標準差
        nn.init.zeros_(self.bias_mu)  # 偏置均值設為零
        nn.init.constant_(self.bias_sigma, self.sigma / np.sqrt(self.in_features))

    def reset_noise(self):
        """重新生成噪聲，保持參數不變。

        原理：
        - 在每次前向傳播或訓練步驟前，重新抽樣噪聲 ε，確保探索行為的隨機性。
        - 噪聲從標準正態分佈 N(0, 1) 中抽樣，與輸入和輸出維度獨立。
        - 在 Pac-Man 中，這確保 DQN 在選擇動作（上、下、左、右）時保持探索多樣性。
        """
        epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device) / np.sqrt(self.in_features)
        epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device) / np.sqrt(self.out_features)
        self.eps_in = torch.sign(epsilon_in) * torch.sqrt(torch.abs(epsilon_in))
        self.eps_out = torch.sign(epsilon_out) * torch.sqrt(torch.abs(epsilon_out))

    def forward(self, x):
        """前向傳播，計算帶噪聲的線性變換。

        原理：
        - 計算公式：y = (μ_w + σ_w * ε_w) * x + (μ_b + σ_b * ε_b)
        - 噪聲矩陣 ε_w 由外積生成：ε_w = eps_out.unsqueeze(1) * eps_in，形狀為 (out_features, in_features)。
        - 在 Pac-Man 中，這將展平的卷積特徵轉換為 Q 值或中間表示，帶有動態噪聲以促進探索。

        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, in_features)。

        Returns:
            torch.Tensor: 輸出張量，形狀為 (batch_size, out_features)。
        """
        x = x.contiguous()  # 確保輸入張量內存連續
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"預期輸入形狀 (batch_size, {self.in_features})，得到 {x.shape}")
        if not hasattr(self, 'eps_in'):
            self.reset_noise()  # 若未初始化噪聲，則生成
        # 計算帶噪聲的權重
        weight_noise = self.eps_out.unsqueeze(1) * self.eps_in  # 噪聲矩陣
        weight = self.weight_mu + self.weight_sigma * weight_noise  # 最終權重
        # 計算帶噪聲的偏置
        bias = self.bias_mu + self.bias_sigma * self.eps_out  # 最終偏置
        weight_sigma_mean = self.weight_sigma.abs().mean().item()
        bias_sigma_mean = self.bias_sigma.abs().mean().item()
        # 計算線性變換：y = x * w^T + b
        return x @ weight.transpose(0, 1) + bias, {"weight_sigma_mean": weight_sigma_mean, "bias_sigma_mean": bias_sigma_mean}

class DQN(nn.Module):
    """深度 Q 學習網絡（DQN），用於逼近 Pac-Man 遊戲的 Q 值函數。

    原理：
    - DQN 通過神經網絡逼近 Q 值函數 Q(s, a)，預測每個動作的期望回報。
    - 採用 Dueling DQN 結構，將 Q 值分解為狀態值 V(s) 和動作優勢 A(s, a)：
      - Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
      - V(s) 表示狀態的總體價值，A(s, a) 表示動作相對平均動作的優勢。
    - 使用卷積層處理空間狀態（例如 6 通道 31x31 的迷宮狀態），NoisyLinear 層增強探索。
    - 在 Pac-Man 中，輸入為 6 通道狀態（迷宮、Pac-Man、鬼魂等），輸出為 4 個動作的 Q 值（上、下、左、右）。

    網絡結構：
    - 卷積層：提取空間特徵（如牆壁、鬼魂位置）。
    - 全連接層：計算 V(s) 和 A(s, a)，輸出 Q 值。
    """
    def __init__(self, state_dim, action_dim, num_conv_layers=3):
        """初始化 DQN 網絡，支援可配置的卷積層數。

        Args:
            state_dim (tuple): 狀態維度，例如 (6, 31, 31)，表示 6 通道的 31x31 迷宮。
            action_dim (int): 動作維度，例如 4（上、下、左、右）。
            num_conv_layers (int): 卷積層數（2、3 或 4），控制模型複雜度。
        """
        super(DQN, self).__init__()
        # 動態構建卷積層
        conv_layers = []
        in_channels = state_dim[0]  # 輸入通道數，例如 6（迷宮、Pac-Man、鬼魂等）
        channels = [16, 32, 64][:num_conv_layers]  # 卷積層的輸出通道數
        h, w = state_dim[1], state_dim[2]  # 初始高度和寬度，例如 31x31
        for i, out_channels in enumerate(channels):
            stride = 2 if i == num_conv_layers - 1 else 1  # 最後一層使用 stride=2 縮減尺寸
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),  # 添加 BatchNorm，穩定訓練並加速收斂
                nn.ReLU(),  # ReLU 激活函數，增加非線性
            ])
            # 更新輸出尺寸，公式：output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1
            h = ((h + 2 * 1 - 3) // stride) + 1
            w = ((w + 2 * 1 - 3) // stride) + 1
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)  # 構建卷積層序列
        conv_out_size = in_channels * h * w  # 計算卷積層展平後的尺寸
        # 全連接層：Dueling DQN 的價值流和優勢流
        self.fc1 = NoisyLinear(conv_out_size, 256)  # 中間層，256 維增強表達能力
        self.fc2_value = NoisyLinear(256, 1)  # 價值流，輸出狀態價值 V(s)
        self.fc2_advantage = NoisyLinear(256, action_dim)  # 優勢流，輸出動作優勢 A(s, a)

    def forward(self, x):
        """前向傳播，計算每個動作的 Q 值。

        原理：
        - 輸入狀態經過卷積層提取空間特徵，展平後輸入全連接層。
        - 使用 Dueling DQN 結構計算 Q 值：
        - V(s)：由 fc2_value 輸出，表示狀態的總體價值。
        - A(s, a)：由 fc2_advantage 輸出，表示每個動作的相對優勢。
        - Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))，通過平均優勢校正穩定輸出。
        - 在 Pac-Man 中，輸入為 (batch_size, 6, 31, 31)，輸出為 (batch_size, 4) 的 Q 值。

        Args:
            x (torch.Tensor): 輸入狀態張量，形狀為 (batch_size, channels, height, width)。

        Returns:
            tuple: (q_values, noise_metrics)
                - q_values: Q 值張量，形狀為 (batch_size, action_dim)。
                - noise_metrics: 字典，包含所有 NoisyLinear 層的噪聲標準差均值。
        """
        x = self.conv(x)  # 通過卷積層提取空間特徵
        batch_size = x.size(0)  # 獲取批量大小
        x = x.contiguous().view(batch_size, -1)  # 展平卷積輸出為 (batch_size, conv_out_size)
        
        # 處理 fc1 的輸出
        x, fc1_metrics = self.fc1(x)  # 分離 NoisyLinear 的輸出和噪聲指標
        x = F.relu(x)  # 僅將 Tensor 傳入 ReLU
        
        # 處理價值流和優勢流
        value, value_metrics = self.fc2_value(x)  # 計算狀態價值 V(s)
        advantage, advantage_metrics = self.fc2_advantage(x)  # 計算動作優勢 A(s, a)
        
        # 合併噪聲指標
        noise_metrics = {
            'fc1_weight_sigma_mean': fc1_metrics['weight_sigma_mean'],
            'fc1_bias_sigma_mean': fc1_metrics['bias_sigma_mean'],
            'value_weight_sigma_mean': value_metrics['weight_sigma_mean'],
            'value_bias_sigma_mean': value_metrics['bias_sigma_mean'],
            'advantage_weight_sigma_mean': advantage_metrics['weight_sigma_mean'],
            'advantage_bias_sigma_mean': advantage_metrics['bias_sigma_mean']
        }
        
        # 計算 Q 值：Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, noise_metrics

    def reset_noise(self):
        """重置所有嘈雜層的噪聲。

        原理：
        - 在每次動作選擇或訓練步驟前，重新生成 NoisyLinear 層的噪聲，確保探索隨機性。
        - 在 Pac-Man 中，這幫助 DQN 在迷宮中嘗試不同路徑，避免陷入局部最優。

        使用：
        - 由 DQNAgent 在選擇動作或更新模型時調用。
        """
        self.fc1.reset_noise()  # 重置中間層噪聲
        self.fc2_value.reset_noise()  # 重置價值流噪聲
        self.fc2_advantage.reset_noise()  # 重置優勢流噪聲