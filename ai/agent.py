# ai/agent.py
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

# --- PER 輔助類 ---
class SumTree:
    """
    SumTree 是一個二叉樹結構，用於高效地儲存優先級並進行採樣。
    """
    def __init__(self, capacity):
        self.capacity = capacity  # 緩衝區容量
        # tree 陣列儲存優先級和總和，大小為 2*capacity - 1
        # 葉節點從 capacity-1 開始，到 2*capacity - 2
        self.tree = np.zeros(2 * capacity - 1)
        # data 陣列儲存實際經驗，大小為 capacity
        self.data = np.array([None] * capacity, dtype=object) # 使用 dtype=object 儲存元組
        self.data_pointer = 0 # 指向下一個可用於儲存經驗的位置

    def add(self, priority, data):
        """
        添加新數據和其優先級到 SumTree 中。
        """
        tree_idx = self.data_pointer + self.capacity - 1 # 計算葉節點在 tree 陣列中的索引
        self.data[self.data_pointer] = data # 儲存實際經驗
        self.update(tree_idx, priority) # 更新葉節點和其父節點的總和

        self.data_pointer += 1
        if self.data_pointer >= self.capacity: # 如果達到容量，從頭開始覆蓋
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        """
        更新樹中某個節點的優先級。
        """
        change = priority - self.tree[tree_idx] # 計算優先級的變化量
        self.tree[tree_idx] = priority # 更新葉節點的優先級
        while tree_idx != 0: # 向上傳播更新，直到根節點
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        從根節點向下查找，根據隨機值 v 選擇一個葉節點。
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree): # 如果是葉節點
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]: # 如果 v 小於等於左子節點的總和，則進入左子節點
                    parent_idx = left_child_idx
                else: # 否則，v 減去左子節點的總和，進入右子節點
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1 # 計算實際經驗在 data 陣列中的索引
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        """
        返回樹的總優先級（根節點的值）。
        """
        return self.tree[0]

# --- DQNAgent 類 ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=100000, batch_size=128, lr=1e-4, epsilon=1.0,
                 gamma=0.99, n_step=1, # 確保 gamma 和 n_step 在這裡定義
                 alpha=0.6, beta=0.4, beta_annealing_steps=200000, # 調整 beta_annealing_steps
                 priority_epsilon=1e-5, target_update_freq=100): # PER 相關參數和目標網路更新頻率
        """
        初始化 DQN 代理，設置 Dueling DQN 模型、記憶緩衝區和訓練參數。
        新增 PER 相關參數：alpha, beta, beta_annealing_steps, priority_epsilon。
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict()) # 初始時目標網路和行為網路相同
        self.target_model.eval() # 目標網路設為評估模式

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.steps = 0 # 總訓練步數
        self.gamma = gamma # 折扣因子
        self.n_step = n_step # n-step 回報 (目前代理內部實現為單步TD，未來可擴展)

        # PER 相關參數
        self.alpha = alpha # 優先級指數
        self.beta = beta # 重要性採樣指數
        self.beta_annealing_steps = beta_annealing_steps # beta 退火步數
        self.priority_epsilon = priority_epsilon # 避免零優先級

        # 使用 SumTree 作為記憶緩衝區
        self.memory = SumTree(buffer_size)
        
        self.target_update_freq = target_update_freq # 目標網路更新頻率

    def update_epsilon(self, min_epsilon=0.01, decay_rate=0.9999):
        """
        隨著訓練進行，逐漸減少探索率 epsilon。
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def update_beta(self):
        """
        更新 IS 權重的 beta 值，從初始值退火到 1。
        """
        self.beta = min(1.0, self.beta + (1.0 - self.beta) / self.beta_annealing_steps)


    def choose_action(self, state):
        """
        基於 epsilon-greedy 策略選擇動作。
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.model.eval() # 設置為評估模式
            with torch.no_grad():
                q_values = self.model(state)
            self.model.train() # 恢復訓練模式
            return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        將經驗存儲到回放緩衝區，並計算初始優先級。
        """
        # PER 的初始優先級通常設為當前緩衝區中最大的優先級，
        # 或者一個預設的最大值，以確保新經驗有機會被採樣。
        # 這裡設置為總優先級的 alpha 次方（即新經驗通常有較高優先級）
        max_priority = self.memory.total_priority ** self.alpha if self.memory.total_priority > 0 else 1.0
        self.memory.add(max_priority, (state, action, reward, next_state, done))


    def sample(self):
        """
        從 SumTree 中採樣一個批次的經驗和它們的 IS 權重。
        """
        batch_memory = []
        priorities = []
        indices = []

        total_p = self.memory.total_priority
        segment = total_p / self.batch_size # 將總優先級分成 batch_size 個區段

        self.update_beta() # 每採樣一次更新 beta

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b) # 在每個 segment 內隨機採樣一個值

            idx, priority, data = self.memory.get_leaf(v)
            batch_memory.append(data)
            priorities.append(priority)
            indices.append(idx)

        # 將數據轉換為適當的張量格式
        # 使用 np.array() 確保數據類型正確
        states, actions, rewards, next_states, dones = zip(*batch_memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # 計算重要性採樣權重
        # N 為緩衝區中實際經驗數量，SumTree 的 data_pointer 或 capacity
        N = self.memory.data_pointer if self.memory.data_pointer > 0 else self.memory.capacity # 如果緩衝區未滿，使用實際數量

        # 計算重要性採樣權重
        # is_weights = ( (N * P(i)) ^ (-beta) ) / max(is_weights)
        # P(i) = priority / total_priority
        min_prob = np.min(np.array(priorities) / total_p) # 最小採樣概率
        max_weight = (min_prob * N)**(-self.beta) # 最大權重，用於歸一化

        is_weights = torch.FloatTensor(
            (np.array(priorities) / total_p * N)**(-self.beta)
        ).unsqueeze(1).to(self.device)
        
        is_weights = is_weights / is_weights.max() # 歸一化，防止權重過大

        return (states, actions, rewards, next_states, dones), indices, is_weights


    def learn(self):
        """
        從記憶緩衝區中採樣並訓練 Dueling DQN 模型。
        這裡實現了 Double DQN 和 PER 的優先級更新。
        """
        # 確保有足夠的經驗進行訓練
        # self.memory.data_pointer 表示當前 SumTree 中實際儲存的經驗數量
        if self.memory.data_pointer < self.batch_size:
            return

        (states, actions, rewards, next_states, dones), indices, is_weights = self.sample()

        # 計算當前 Q 值: Q(s,a;theta)
        q_values = self.model(states).gather(1, actions)

        # 計算目標 Q 值: Double DQN
        # 使用行為網路選擇下一個狀態的最佳動作 a*
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        # 使用目標網路評估這個最佳動作的 Q 值 Q(s',a*;theta_target)
        next_q_values = self.target_model(next_states).gather(1, next_actions).detach()

        # 計算目標 Q 值 for Bellman equation (單步 TD)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 計算 TD 誤差，用於更新優先級
        # 確保 td_errors 是 numpy 陣列，方便後續 SumTree 操作
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy().flatten()

        # 更新採樣經驗的優先級
        for i, idx in enumerate(indices):
            self.memory.update(idx, td_errors[i] ** self.alpha + self.priority_epsilon) # 應用 alpha 冪次

        # 計算損失
        # PER: 損失函數 = IS_weights * MSE(Q, Q_target)
        # reduction='none' 使得 F.mse_loss 返回每個樣本的損失，然後再應用 IS 權重
        loss = (is_weights * F.mse_loss(q_values, target_q_values, reduction='none')).mean()


        # 反向傳播和優化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（可選，但推薦用於穩定訓練）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        
        # 定期軟更新目標網路
        if self.steps % self.target_update_freq == 0:
            self.soft_update_target()
            
        return loss.item()

    def soft_update_target(self, tau=0.005):
        """
        軟更新目標網路的權重。
        tau (float): 更新率。
        """
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * model_param.data + (1.0 - tau) * target_param.data)

    def save(self, path, memory_path=None):
        """
        保存模型權重和記憶緩衝區。
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'beta': self.beta # 保存 beta
        }
        torch.save(checkpoint, path)
        if memory_path:
            # 對於 SumTree，需要保存 tree 和 data
            with open(memory_path, 'wb') as f:
                # 使用 copy() 確保保存的是數據的副本，而不是視圖
                # data_pointer 也需要保存
                pickle.dump({'tree': self.memory.tree.copy(), 
                             'data': self.memory.data.copy(), 
                             'data_pointer': self.memory.data_pointer}, f)

    def load(self, path, memory_path=None):
        """
        載入模型權重和記憶緩衝區。
        """
        checkpoint = torch.load(path, map_location=self.device) # 確保在正確的設備上載入
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.beta = checkpoint.get('beta', self.beta) # 載入 beta，如果不存在則使用默認值
        self.soft_update_target() # 載入後同步目標網路

        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory_data = pickle.load(f)
                # 載入 SumTree 的狀態
                self.memory.tree = memory_data['tree']
                self.memory.data = memory_data['data']
                self.memory.data_pointer = memory_data['data_pointer']