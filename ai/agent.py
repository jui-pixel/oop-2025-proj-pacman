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
            epsilon (float): 初始探索率，預設為 1.2（增加探索）。
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = 0.995
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9997
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.steps = 0
        self.target_update_freq = 500
        self.alpha = 0.25
        self.beta = 0.4
        self.beta_increment = 0.002
        self.last_action = None
        self.last_position = None
        self.stuck_counter = 0
        self.update_target_model()
        self.action_cooldown = 0  # 添加動作冷卻計數器
        self.cooldown_steps = 10  # 每 10 步執行一次動作

    def update_target_model(self):
        """
        將主模型的權重複製到目標模型，確保兩者一致。
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def _is_valid_action(self, state, action):
        """
        檢查動作是否有效（是否會撞牆）。

        Args:
            state (numpy.ndarray): 當前狀態，形狀為 (height, width, channels)。
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。

        Returns:
            bool: 動作是否有效。
        """
        pacman_x, pacman_y = np.where(state[:, :, 0] == 1.0)
        if len(pacman_x) == 0:
            return True
        pacman_x, pacman_y = pacman_x[0], pacman_y[0]

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x, new_y = pacman_x + dx, pacman_y + dy

        if 0 <= new_x < state.shape[0] and 0 <= new_y < state.shape[1]:
            return state[new_x, new_y, 5] != 1.0
        return False

    def _check_stuck(self, state):
        """
        檢查 Pac-Man 是否停滯（位置未改變）。

        Args:
            state (numpy.ndarray): 當前狀態。

        Returns:
            bool: 是否停滯。
        """
        pacman_x, pacman_y = np.where(state[:, :, 0] == 1.0)
        if len(pacman_x) == 0:
            return False
        current_position = (pacman_x[0], pacman_y[0])

        if self.last_position is not None and current_position == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = current_position
        return self.stuck_counter >= 2

    def get_action(self, state):
        """
        使用改進的 ε-貪婪策略選擇動作，確保移動並增加轉向（先挑選 valid_actions）。

        Args:
            state (numpy.ndarray): 當前狀態，形狀為 (height, width, channels)。

        Returns:
            int: 選擇的動作索引（0: 上, 1: 下, 2: 左, 3: 右）。
        """
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            return self.last_action if self.last_action is not None else random.randrange(self.action_dim)
        # 先挑選所有有效動作
        valid_actions = [a for a in range(self.action_dim) if self._is_valid_action(state, a)]
        if not valid_actions:
            self.action_cooldown = self.cooldown_steps
            self.last_action = random.randrange(self.action_dim)
            return self.last_action

        is_stuck = self._check_stuck(state)

        if random.random() < self.epsilon or is_stuck:
            self.action_cooldown = self.cooldown_steps
            self.last_action = random.randrange(self.action_dim)
            return self.last_action  # 隨機選擇有效動作

        # 最大 Q 值選取，限制在 valid_actions
        with torch.no_grad():  # 禁用梯度計算以節省資源
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)  # 計算 Q 值
            self.action_cooldown = self.cooldown_steps
            self.last_action = q_values.argmax().item()  # 選擇最大 Q 值的動作
            return self.last_action

    def train(self):
        """
        從記憶緩衝區採樣並訓練主模型，使用優先級經驗回放和重要性採樣。

        Returns:
            float: 當前批次的損失值，若記憶不足則返回 0。
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        batch = [self.memory[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = (q_values - target_q_values).abs().cpu().detach().numpy()
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error + 1e-5

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        loss = (weights * nn.MSELoss(reduction='none')(q_values, target_q_values)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()
        self.beta = min(1.0, self.beta + self.beta_increment)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        max_priority = max(self.priorities) if self.priorities else 1.0
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
        self.update_target_model()
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory = pickle.load(f)
                self.memory = deque(memory, maxlen=self.memory.maxlen)