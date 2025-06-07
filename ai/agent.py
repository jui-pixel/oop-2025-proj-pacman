import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
import os
from collections import deque

class DQN(nn.Module):
    """標準 DQN 神經網路，輸入狀態，輸出動作 Q 值。"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        conv_out_size = state_dim[1] * state_dim[2] * 64
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=50000, batch_size=64, 
                 lr=2.5e-4, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50000, 
                 gamma=0.99, target_update_freq=200):
        """
        初始化基礎 DQN 代理。

        Args:
            state_dim (tuple): 狀態維度 (channels, height, width)。
            action_dim (int): 動作數量。
            device (str): 訓練設備 ('cpu' 或 'cuda')。
            buffer_size (int): 回放緩衝區大小。
            batch_size (int): 批次大小。
            lr (float): 學習率。
            epsilon_start (float): 初始 epsilon（探索率）。
            epsilon_end (float): 最終 epsilon。
            epsilon_decay_steps (int): epsilon 衰減步數。
            gamma (float): 折扣因子。
            target_update_freq (int): 目標網路更新頻率。
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

    def update_epsilon(self):
        """更新 epsilon，線性衰減。"""
        decay_progress = min(self.steps / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress

    def choose_action(self, state):
        """根據 epsilon-greedy 策略選擇動作。"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()
        return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """儲存經驗到回放緩衝區。"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """從回放緩衝區均勻採樣批次。"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def learn(self):
        """執行一次 DQN 學習步驟。"""
        if len(self.memory) < self.batch_size:
            return None

        self.steps += 1
        self.update_epsilon()

        states, actions, rewards, next_states, dones = self.sample()
        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save(self, model_path, memory_path):
        """保存模型和回放緩衝區。"""
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(self.memory, f)
        print(f"Saved model to {model_path} and memory to {memory_path}")

    def load(self, model_path, memory_path=None):
        """載入模型和回放緩衝區。"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Loaded memory from {memory_path}")
        print(f"Loaded model from {model_path}")