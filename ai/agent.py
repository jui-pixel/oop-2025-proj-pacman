"""
定義 DQN 代理，負責管理記憶緩衝區、動作選擇和模型訓練。
支援 Dueling DQN、優先級經驗回放 (PER)、n-step 學習、軟更新和 Double DQN。
此代理實現了深度強化學習的核心邏輯，優化 Pac-Man 遊戲策略。
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from ai.dqn import DuelingDQN
import pickle

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=100000, batch_size=128, lr=1e-4, epsilon=0.9):
        """
        初始化 DQN 代理，設置 Dueling DQN 模型、記憶緩衝區和訓練參數。

        Args:
            state_dim (Tuple[int, int, int]): 狀態維度 (高度, 寬度, 通道數)，例如 (31, 28, 6)。
            action_dim (int): 動作數量，例如 4（上、下、左、右）。
            device (str): 計算設備，"cuda" 或 "cpu"，預設為 "cpu"。
            buffer_size (int): 記憶緩衝區的最大容量，預設為 100000。
            batch_size (int): 每次訓練的批次大小，預設為 128。
            lr (float): 學習率，預設為 1e-4。
            epsilon (float): 初始探索率，預設為 0.9。
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.gamma = 0.995  # 折扣因子
        self.n_step = 3  # n-step 學習步長
        self.tau = 0.005  # 軟更新因子

        # 初始化 Dueling DQN 模型
        self.model = DuelingDQN(state_dim, action_dim).to(device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.priorities = deque(maxlen=buffer_size)

        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.alpha = 0.6  # 優先級經驗回放的 alpha 參數
        self.beta = 0.4  # 重要性採樣的 beta 參數
        self.beta_increment = 1e-3
        self.steps = 0

        # 保留原有的動作控制和停滯檢查變數
        self.last_action = None
        self.last_position = None
        self.stuck_counter = 0
        self.action_cooldown = 0
        self.cooldown_steps = 0
        self.recent_rewards = deque(maxlen=100)

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
        檢查 Pac-Man 是否停滯。

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
        使用 ε-貪婪策略選擇動作，結合動作冷卻和停滯檢查。

        Args:
            state (numpy.ndarray): 當前狀態。

        Returns:
            int: 選擇的動作索引。
        """
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            return self.last_action if self.last_action is not None else random.randrange(self.action_dim)

        valid_actions = [a for a in range(self.action_dim) if self._is_valid_action(state, a)]
        if not valid_actions:
            self.action_cooldown = self.cooldown_steps
            return random.randrange(self.action_dim)

        is_stuck = self._check_stuck(state)

        if random.random() < self.epsilon or is_stuck:
            base_weights = [1.0] * self.action_dim
            if self.last_action is not None:
                if self.last_action == 0:
                    weights = [1.7, 0.3, 1.0, 1.0]
                elif self.last_action == 1:
                    weights = [0.3, 1.7, 1.0, 1.0]
                elif self.last_action == 2:
                    weights = [1.0, 1.0, 1.7, 0.3]
                elif self.last_action == 3:
                    weights = [1.0, 1.0, 0.3, 1.7]
            else:
                weights = base_weights

            action_weights = [weights[a] for a in valid_actions]
            if sum(action_weights) == 0:
                action_weights = [1.0] * len(valid_actions)
            else:
                action_weights = [w / sum(action_weights) for w in action_weights]

            action = random.choices(valid_actions, weights=action_weights, k=1)[0]
        else:
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
            q_values_valid = [(q_values_np[a], a) for a in valid_actions]
            if not q_values_valid:
                action = random.choice(valid_actions)
            else:
                action = max(q_values_valid, key=lambda x: x[0])[1]

        self.last_action = action
        self.action_cooldown = self.cooldown_steps
        return action

    def _get_n_step_info(self):
        """
        計算 n-step 回報和對應的狀態轉換。

        Returns:
            Tuple: (state, action, reward, next_state, done)。
        """
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        state, action = self.n_step_buffer[0][:2]
        return state, action, reward, next_state, done

    def remember(self, state, action, reward, next_state, done):
        """
        儲存經驗，支援 n-step 學習。

        Args:
            state (numpy.ndarray): 當前狀態。
            action (int): 執行的動作。
            reward (float): 獲得的獎勵。
            next_state (numpy.ndarray): 下一個狀態。
            done (bool): 是否結束。
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        transition = self._get_n_step_info()
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(transition)
        self.priorities.append(max_priority)

    def soft_update_target(self):
        """
        軟更新目標模型參數。
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self):
        """
        從記憶緩衝區採樣並訓練模型，使用優先級經驗回放、Double DQN 和 n-step 學習。

        Returns:
            float: 當前批次的損失值。
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        td_errors = (q_values - targets).abs().detach().cpu().numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_errors[i] + 1e-5

        loss = (F.mse_loss(q_values, targets, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target()
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.steps += 1

        return loss.item()

    def save(self, path, memory_path=None):
        """
        保存模型權重和記憶緩衝區。

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
        載入模型權重和記憶緩衝區。

        Args:
            path (str): 模型檔案路徑。
            memory_path (str, optional): 記憶緩衝區檔案路徑。
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.soft_update_target()
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory = pickle.load(f)
                self.memory = deque(memory, maxlen=self.memory.maxlen)
    
    def update_epsilon(self):
        """
        更新 ε-貪婪策略中的探索率。
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)