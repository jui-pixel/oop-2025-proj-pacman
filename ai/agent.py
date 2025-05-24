# ai/agent.py
"""
定義 DQN 代理，負責管理記憶緩衝區、動作選擇和模型訓練。
支援模型保存和載入，實現深度強化學習的訓練邏輯。
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
        初始化 DQN 代理，設置模型、記憶緩衝區和訓練參數。

        Args:
            state_dim (Tuple[int, int, int]): 狀態維度 (高度, 寬度, 通道數)。
            action_dim (int): 動作數量。
            device (str): 計算設備，預設為 "cpu"。
            buffer_size (int): 記憶緩衝區大小，預設為 10000。
            batch_size (int): 訓練批次大小，預設為 64。
            lr (float): 學習率，預設為 5e-4。
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.memory = deque(maxlen=buffer_size)  # 記憶緩衝區
        self.batch_size = batch_size
        self.gamma = 0.99  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.9999  # 探索率衰減率
        self.model = DQN(state_dim, action_dim).to(device)  # 主模型
        self.target_model = DQN(state_dim, action_dim).to(device)  # 目標模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.steps = 0  # 訓練步數
        self.target_update_freq = 1000  # 目標模型更新頻率
        self.update_target_model()  # 同步初始模型

    def update_target_model(self):
        """將主模型的權重複製到目標模型。"""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        """
        根據當前狀態選擇動作，使用 ε-貪婪策略。

        Args:
            state (numpy.ndarray): 當前狀態。

        Returns:
            int: 選擇的動作索引。
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # 隨機探索
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()  # 選擇 Q 值最大的動作

    def train(self):
        """
        從記憶緩衝區採樣並訓練模型。

        Returns:
            float: 當前批次的損失值，若記憶不足則返回 0。
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 隨機採樣批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 轉換為張量
        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 計算當前 Q 值
        q_values = self.model(states).gather(1, actions).squeeze(1)
        # 計算目標 Q 值
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 計算損失
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        # 定期更新目標模型
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()
        
        # 衰減探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def remember(self, state, action, reward, next_state, done):
        """
        將經驗儲存到記憶緩衝區。

        Args:
            state (numpy.ndarray): 當前狀態。
            action (int): 執行的動作。
            reward (float): 獲得的獎勵。
            next_state (numpy.ndarray): 下一個狀態。
            done (bool): 是否結束。
        """
        self.memory.append((state, action, reward, next_state, done))

    def save(self, path, memory_path=None):
        """
        保存模型權重、優化器狀態、探索率和記憶緩衝區。

        Args:
            path (str): 模型檔案路徑。
            memory_path (str, optional): 記憶緩衝區檔案路徑。
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
            path (str): 模型檔案路徑。
            memory_path (str, optional): 記憶緩衝區檔案路徑。
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.update_target_model()
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory = pickle.load(f)
                self.memory = deque(memory, maxlen=self.memory.maxlen)