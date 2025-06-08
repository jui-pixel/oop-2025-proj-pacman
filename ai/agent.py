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

# 定義一個名為 Transition 的命名元組，用來儲存強化學習中的轉換數據（狀態、動作、獎勵、下個狀態、是否結束）
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=100000, batch_size=128, 
                 lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=1000000, 
                 gamma=0.99, target_update_freq=200, n_step=4, alpha=0.6, beta=0.4, 
                 beta_increment=0.001, expert_prob_start=0.3, expert_prob_end=0.0, 
                 expert_prob_decay_steps=500000):
        """
        初始化 DQN 代理，設定深度強化學習的參數。

        Args:
            state_dim (tuple): 狀態空間的維度，例如 (6, 21, 21)。
            action_dim (int): 動作空間的維度，例如 4（上、下、左、右）。
            device (str): 計算設備，預設為 "cpu" 或 "cuda"（若有 GPU）。
            buffer_size (int): 回放緩衝區的最大容量。
            batch_size (int): 每次訓練從回放緩衝區採樣的數據量。
            lr (float): 學習率，控制模型參數更新的步幅。
            epsilon_start (float): 探索率初始值，決定隨機動作的機率。
            epsilon_end (float): 探索率的最小值。
            epsilon_decay_steps (int): 探索率衰減的步數。
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
        self.epsilon = epsilon_start  # 當前探索率
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.expert_prob = expert_prob_start  # 當前專家策略機率
        self.expert_prob_start = expert_prob_start
        self.expert_prob_end = expert_prob_end
        self.expert_prob_decay_steps = expert_prob_decay_steps
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.alpha = alpha  # 控制優先級的大小
        self.beta = beta    # 控制重要性採樣的權重
        self.beta_increment = beta_increment
        self.steps = 0  # 記錄訓練步數

        # 初始化主模型和目標模型
        self.model = DQN(state_dim, action_dim).to(self.device)  # 主網絡用於預測 Q 值
        self.target_model = DQN(state_dim, action_dim).to(self.device)  # 目標網絡穩定訓練
        self.target_model.load_state_dict(self.model.state_dict())  # 複製主模型參數到目標模型
        self.target_model.eval()  # 設置為評估模式，不更新梯度

        # 設置優化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # 初始化優先級回放緩衝區
        self.memory = SumTree(buffer_size)
        # n-step 回放緩衝區
        self.n_step_memory = deque(maxlen=n_step)
        self.max_priority = 1.0  # 最大優先級，初始化為 1

    def update_epsilon(self):
        """根據訓練步數更新探索率，逐步從 epsilon_start 衰減到 epsilon_end。"""
        decay_progress = min(self.steps / self.epsilon_decay_steps, 1.0)  # 計算衰減進度
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
        # 更新專家策略機率
        self.expert_prob = self.expert_prob_start - (self.expert_prob_start - self.expert_prob_end) * decay_progress

    def choose_action(self, state):
        """
        根據 ε-貪婪策略選擇動作。

        Args:
            state (array): 當前狀態。

        Returns:
            int: 選擇的動作索引。
        """
        if random.random() < self.epsilon:  # 以 epsilon 機率隨機選擇動作（探索）
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 將狀態轉為張量
        self.model.eval()  # 切換到評估模式
        with torch.no_grad():  # 不計算梯度
            q_values = self.model(state)  # 獲取所有動作的 Q 值
        self.model.train()  # 切回訓練模式
        return q_values.argmax(1).item()  # 選擇 Q 值最大的動作

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
        self.n_step_memory.append((state, action, reward, next_state, done))
        # 當 n-step 緩衝區滿或遊戲結束時，計算 n-step 獎勵並儲存
        if len(self.n_step_memory) == self.n_step or done:
            total_reward = 0
            final_state = next_state
            final_done = done
            for i, (s, a, r, ns, d) in enumerate(reversed(self.n_step_memory)):
                total_reward = r + self.gamma * total_reward  # 計算 n-step 折扣獎勵
                if i == 0:
                    final_state = s
                if d:
                    break
            transition = Transition(final_state, a, total_reward, next_state, done)
            priority = self.max_priority if not self.memory.total_priority else self.max_priority
            self.memory.add(priority, transition)  # 將轉換加入優先級回放緩衝區
            if done:
                self.n_step_memory.clear()

    def sample(self):
        """
        從優先級回放緩衝區中採樣一批數據。

        Returns:
            tuple: 狀態、動作、獎勵、下個狀態、是否結束、重要性採樣權重、索引。
        """
        if self.memory.total_priority == 0:  # 緩衝區為空
            return None, None, None, None, None, None, None

        indices = []
        batch = []
        weights = []
        segment = self.memory.total_priority / self.batch_size  # 每個分段的優先級
        self.beta = min(1.0, self.beta + self.beta_increment)  # 增加重要性採樣權重

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)  # 在分段內隨機選擇
            idx, priority, data = self.memory.get_leaf(s)  # 獲取數據
            indices.append(idx)
            batch.append(data)
            prob = priority / self.memory.total_priority  # 計算機率
            weight = (self.memory.capacity * prob) ** (-self.beta)  # 計算重要性採樣權重
            weights.append(weight)

        # 轉換為張量
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
            # 隨機採樣一批專家數據
            batch = random.sample(expert_data, min(self.batch_size, len(expert_data)))
            states, actions = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)

            # 計算交叉熵損失
            q_values = self.model(states)
            loss = F.cross_entropy(q_values, actions)

            # 優化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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

        self.steps += 1
        self.update_epsilon()  # 更新探索率
        states, actions, rewards, next_states, dones, weights, indices = self.sample()
        q_values = self.model(states).gather(1, actions)  # 獲取當前 Q 值

        with torch.no_grad():  # 不計算梯度
            next_actions = self.model(next_states).max(1, keepdim=True)[1]  # 選擇下個狀態的最佳動作
            next_q_values = self.target_model(next_states).gather(1, next_actions)  # 獲取目標 Q 值
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values  # 計算目標值

        td_errors = (q_values - target_q_values).abs()  # 計算 TD 誤差
        loss = (td_errors * weights).mean()  # 加權損失

        self.optimizer.zero_grad()  # 清除之前梯度
        loss.backward()  # 反向傳播
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 裁剪梯度防止爆炸
        self.optimizer.step()  # 更新參數

        # 更新優先級
        for i, idx in enumerate(indices):
            priority = td_errors[i].detach().cpu().numpy()[0] + 1e-6  # 避免零優先級
            self.memory.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

        if self.steps % self.target_update_freq == 0:  # 定期更新目標網絡
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save(self, model_path, memory_path):
        """將模型和記憶數據保存到文件。"""
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(list(self.memory.data), f)  # 儲存回放緩衝區數據
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