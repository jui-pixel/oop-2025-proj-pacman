# ai/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 將父目錄加入路徑以便導入其他模塊
from collections import deque, namedtuple
from ai.dqn import DQN, NoisyLinear
from ai.sumtree import SumTree

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=50000, batch_size=128, 
                 lr=1e-3, gamma=0.99, target_update_freq=1000, n_step=4, alpha=0.6, beta=0.4, 
                 beta_increment=0.001, expert_prob_start=0.3, expert_prob_end=0.0, 
                 expert_prob_decay_steps=200000):
        """
        初始化 DQN 代理，設定深度強化學習的參數。

        Args:
            state_dim (tuple): 狀態空間的維度，例如 (6, 21, 21)。
            action_dim (int): 動作空間的維度，例如 4（上、下、左、右）。
            device (str): 計算設備，預設為 "cpu" 或 "cuda"。
            buffer_size (int): 回放緩衝區的最大容量。
            batch_size (int): 每次訓練從回放緩衝區採樣的數據量。
            lr (float): 學習率，控制模型參數更新的步幅。
            gamma (float): 折扣因子，衡量未來獎勵的重要性。
            target_update_freq (int): 目標網絡更新頻率（步數）。
            n_step (int): n-step 回放的步數，增強獎勵計算。
            alpha (float): 優先級回放的優先級指數。
            beta (float): 重要性採樣的初始權重。
            beta_increment (float): 每次迭代增加的重要性採樣權重。
            expert_prob_start (float): 初始專家策略機率。
            expert_prob_end (float): 最終專家策略機率。
            expert_prob_decay_steps (int): 專家策略機率衰減步數。
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.expert_prob = expert_prob_start
        self.expert_prob_start = expert_prob_start
        self.expert_prob_end = expert_prob_end
        self.expert_prob_decay_steps = expert_prob_decay_steps
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.steps = 0
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = SumTree(buffer_size)
        self.n_step_memory = deque(maxlen=n_step)
        self.max_priority = 1.0

    def update_expert_prob(self):
        """更新專家策略機率，逐步從 expert_prob_start 衰減到 expert_prob_end。"""
        decay_progress = min(self.steps / self.expert_prob_decay_steps, 1.0)
        self.expert_prob = self.expert_prob_start - (self.expert_prob_start - self.expert_prob_end) * decay_progress

    def choose_action(self, state):
        """
        根據當前策略選擇動作（Noisy DQN 無需 ε-貪婪）。

        Args:
            state (array): 當前狀態。

        Returns:
            int: 選擇的動作索引。
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()
        return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        儲存轉換數據到 n-step 回放緩衝區。

        Args:
            state (array): 當前狀態。
            action (int): 執行的動作。
            reward (float): 獲得的獎勵。
            next_state (array): 下個狀態。
            done (bool): 是否結束。
        """
        if not isinstance(state, np.ndarray) or state.shape != self.state_dim:
            raise ValueError(f"Invalid state shape: expected {self.state_dim}, got {state.shape}")
        if not isinstance(next_state, np.ndarray) or next_state.shape != self.state_dim:
            raise ValueError(f"Invalid next_state shape: expected {self.state_dim}, got {next_state.shape}")
        if not (0 <= action < self.action_dim):
            raise ValueError(f"Invalid action: {action}, expected 0 to {self.action_dim-1}")

        self.n_step_memory.append((state, action, reward, next_state, done))
        if len(self.n_step_memory) >= self.n_step or done:
            for i in range(len(self.n_step_memory) - (1 if done else 0)):
                total_reward = 0
                final_state = self.n_step_memory[i][0]
                final_action = self.n_step_memory[i][1]
                final_next_state = next_state
                final_done = done
                for j in range(i, min(i + self.n_step, len(self.n_step_memory))):
                    r = self.n_step_memory[j][2]
                    total_reward += (self.gamma ** (j - i)) * r
                    if j == len(self.n_step_memory) - 1:
                        final_next_state = self.n_step_memory[j][3]
                        final_done = self.n_step_memory[j][4]
                    if self.n_step_memory[j][4]:
                        final_done = True
                        break
                total_reward = total_reward / 100.0
                transition = Transition(final_state, final_action, total_reward, final_next_state, final_done)
                priority = self.max_priority + 1e-6
                self.memory.add(priority, transition)
            if done:
                self.n_step_memory.clear()
            elif len(self.n_step_memory) >= self.n_step:
                self.n_step_memory.popleft()

    def sample(self):
        """
        從優先級回放緩衝區中採樣一批數據。

        Returns:
            tuple: 狀態、動作、獎勵、下個狀態、是否結束、重要性採樣權重、索引。
        """
        if self.memory.total_priority == 0:
            return None, None, None, None, None, None, None
        indices = []
        batch = []
        weights = []
        segment = self.memory.total_priority / self.batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.memory.get_leaf(s)
            indices.append(idx)
            batch.append(data)
            prob = priority / self.memory.total_priority
            weight = (self.memory.capacity * prob) ** (-self.beta)
            weights.append(weight)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1) / max(weights)
        return states, actions, rewards, next_states, dones, weights, indices

    def pretrain(self, expert_data, pretrain_steps=1000):
        """
        使用專家數據進行模仿學習預訓練。

        Args:
            expert_data (list): 包含 (state, action) 的專家數據。
            pretrain_steps (int): 預訓練步數。
        """
        print(f"Starting pretraining with {len(expert_data)} expert samples for {pretrain_steps} steps...")
        self.model.train()
        for step in range(pretrain_steps):
            batch = random.sample(expert_data, min(self.batch_size, len(expert_data)))
            states, actions = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).squeeze()
            q_values = self.model(states)
            loss = F.cross_entropy(q_values, actions)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            if (step + 1) % 100 == 0:
                print(f"Pretrain step {step + 1}/{pretrain_steps}, Loss: {loss.item():.4f}")
        print("Pretraining completed")

    def learn(self, expert_action=False):
        """
        執行一次學習步驟，更新模型參數。

        Args:
            expert_action (bool): 是否為專家動作。

        Returns:
            float: 損失值，若無足夠數據則返回 None。
        """
        if self.memory.total_priority == 0 or len(self.memory.data) < self.batch_size:
            return None
        if expert_action:
            return None
        self.steps += 1
        self.update_expert_prob()
        states, actions, rewards, next_states, dones, weights, indices = self.sample()
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.model(next_states).max(1, keepdim=True)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
        td_errors = (q_values - target_q_values).abs()
        loss = (td_errors * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        for i, idx in enumerate(indices):
            priority = td_errors[i].detach().cpu().numpy()[0] + 1e-6
            self.memory.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
        if self.steps % self.target_update_freq == 0:
            tau = 0.005
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        return loss.item()

    def save(self, model_path, memory_path):
        """將模型和記憶數據保存到文件。"""
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(list(self.memory.data), f)
        print(f"Saved model to {model_path} and memory to {memory_path}")

    def load(self, model_path, memory_path=None):
        """從文件載入模型和記憶數據（若提供）。"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                data = pickle.load(f)
                for d in data:
                    if d is not None:
                        self.memory.add(self.max_priority, d)
            print(f"Loaded memory from {memory_path}")
        print(f"Loaded model from {model_path}")