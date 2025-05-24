# ai/agent.py
"""
定義 DQN 代理，負責管理記憶緩衝區、動作選擇和模型訓練。
支援優先級經驗回放 (PER)、模型保存和載入，實現深度強化學習的核心邏輯。
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from ai.dqn import DQN
import pickle

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=10000, batch_size=64, lr=5e-4, epsilon=1.0):
        """
        初始化 DQN 代理，設置主模型、目標模型、記憶緩衝區和訓練參數。

        Args:
            state_dim (Tuple[int, int, int]): 狀態維度 (高度, 寬度, 通道數)，例如 (31, 28, 6)。
            action_dim (int): 動作數量，例如 4（上、下、左、右）。
            device (str): 計算設備，"cuda" 或 "cpu"，預設為 "cpu"。
            buffer_size (int): 記憶緩衝區的最大容量，預設為 10000。
            batch_size (int): 每次訓練的批次大小，預設為 64。
            lr (float): 學習率，預設為 5e-4。
            epsilon (float): 初始探索率，預設為 1.0（完全隨機）。
        """
        self.state_dim = state_dim  # 儲存狀態維度
        self.action_dim = action_dim  # 儲存動作數量
        self.device = device  # 計算設備
        self.memory = deque(maxlen=buffer_size)  # 記憶緩衝區，儲存經驗
        self.priorities = deque(maxlen=buffer_size)  # 儲存優先級（用於 PER）
        self.batch_size = batch_size  # 訓練批次大小
        self.gamma = 0.995  # 折扣因子，衡量未來獎勵的重要性
        self.epsilon = epsilon  # 探索率，控制隨機動作的概率
        self.epsilon_min = 0.01  # 最小探索率，確保長期仍有探索
        self.epsilon_decay = 0.995  # 探索率衰減率，逐漸減少隨機性
        self.model = DQN(state_dim, action_dim).to(device)  # 主模型，用於動作選擇和訓練
        self.target_model = DQN(state_dim, action_dim).to(device)  # 目標模型，穩定 Q 值計算
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Adam 優化器
        self.steps = 0  # 記錄訓練步數
        self.target_update_freq = 1000  # 目標模型更新頻率（步數）
        self.alpha = 0.4  # 優先級權重，控制 PER 採樣偏見
        self.beta = 0.4  # 重要性採樣權重，初始值
        self.beta_increment = 0.002  # 重要性採樣權重增量
        self.update_target_model()  # 同步初始模型權重

    def update_target_model(self):
        """
        將主模型的權重複製到目標模型，確保兩者一致。

        這是 DQN 的核心機制，通過定期更新目標模型穩定訓練。
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        """
        使用 ε-貪婪策略根據當前狀態選擇動作。

        Args:
            state (numpy.ndarray): 當前狀態，形狀為 (height, width, channels)。

        Returns:
            int: 選擇的動作索引（0: 上, 1: 下, 2: 左, 3: 右）。
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # 隨機探索
        with torch.no_grad():  # 禁用梯度計算以節省資源
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)  # 計算 Q 值
            return q_values.argmax().item()  # 選擇最大 Q 值的動作

    def train(self):
        """
        從記憶緩衝區採樣並訓練主模型，使用優先級經驗回放和重要性採樣。

        Returns:
            float: 當前批次的損失值，若記憶不足則返回 0。
        """
        if len(self.memory) < self.batch_size:
            return 0.0  # 記憶不足，無法訓練

        # 計算優先級概率（基於 TD 誤差）
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        batch = [self.memory[i] for i in indices]

        # 將批次數據轉換為張量
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 計算 Q 值和目標 Q 值
        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 更新優先級（基於 TD 誤差）
        td_errors = (q_values - target_q_values).abs().cpu().detach().numpy()
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error + 1e-5  # 添加小值避免零優先級

        # 計算重要性採樣權重
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        # 計算加權損失並反向傳播
        loss = (weights * nn.MSELoss(reduction='none')(q_values, target_q_values)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新訓練步數和參數
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()  # 定期更新目標模型
        self.beta = min(1.0, self.beta + self.beta_increment)  # 遞增 beta
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 衰減探索率

        return loss.item()

    def remember(self, state, action, reward, next_state, done):
        """
        將經驗儲存到記憶緩衝區，並為新經驗設置最高優先級。

        Args:
            state (numpy.ndarray): 當前狀態。
            action (int): 執行的動作。
            reward (float): 獲得的獎勵。
            next_state (numpy.ndarray): 下一個狀態。
            done (bool): 是否結束。
        """
        max_priority = max(self.priorities) if self.priorities else 1.0  # 新經驗給最高優先級
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def save(self, path, memory_path=None):
        """
        保存模型權重、優化器狀態、探索率和記憶緩衝區。

        Args:
            path (str): 模型檔案路徑（.pth）。
            memory_path (str, optional): 記憶緩衝區檔案路徑（.pkl）。
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        torch.save(checkpoint, path)
        if memory_path:
            with open(memory_path, 'wb') as f:
                pickle.dump(list(self.memory), f)

    def load(self, path, memory_path=None):
        """
        載入模型權重、優化器狀態、探索率和記憶緩衝區。

        Args:
            path (str): 模型檔案路徑（.pth）。
            memory_path (str, optional): 記憶緩衝區檔案路徑（.pkl）。
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.update_target_model()  # 同步目標模型
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory = pickle.load(f)
                self.memory = deque(memory, maxlen=self.memory.maxlen)