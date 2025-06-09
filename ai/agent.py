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
from torch.amp import autocast, GradScaler
# 定義轉換元組，用於儲存強化學習中的單次轉換數據
# 包含：當前狀態、執行的動作、獲得的獎勵、下一個狀態、是否終止
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=100000, batch_size=128, 
                 lr=5e-4, gamma=0.95, target_update_freq=500, n_step=8, alpha=0.6, beta=0.4, 
                 beta_increment=0.001, expert_prob_start=0.3, expert_prob_end=0.01, 
                 expert_prob_decay_steps=200000):
        """
        初始化 DQN 代理，設定深度強化學習的參數。

        原理：
        - DQN（Deep Q-Network）是一種深度強化學習算法，結合 Q-Learning 和深度神經網絡，通過學習 Q 值函數來選擇最優動作。
        - 本實現包含以下進階技術：
          1. **Noisy DQN**：使用 NoisyLinear 層引入隨機性，替代 ε-貪婪策略，促進探索。
          2. **n-step 學習**：計算未來 n 步的累積獎勵，增強長期獎勵的影響。
          3. **優先級經驗回放（Prioritized Experience Replay）**：根據 TD 誤差優先採樣重要轉換，提高學習效率。
          4. **模仿學習預訓練**：使用專家數據初始化模型，加速收斂。
          5. **軟目標更新**：以平滑的方式更新目標網絡，穩定訓練。
        - Q 值公式：Q(s, a) = r + γ * max(Q(s', a'))，其中：
          - s：當前狀態
          - a：執行動作
          - r：即時獎勵
          - s'：下一狀態
          - γ：折扣因子
          - max(Q(s', a'))：下一狀態的最大 Q 值

        Args:
            state_dim (tuple): 狀態空間維度，例如 (6, 21, 21)，表示 6 通道的 21x21 格子。
            action_dim (int): 動作空間大小，例如 4（上、下、左、右）。
            device (str): 計算設備（"cpu" 或 "cuda"）。
            buffer_size (int): 經驗回放池的最大容量。
            batch_size (int): 每次訓練的批量大小。
            lr (float): 學習率，控制參數更新速度。
            gamma (float): 折扣因子，衡量未來獎勵的重要性（0 ≤ γ < 1）。
            target_update_freq (int): 目標網絡更新的頻率（步數）。
            n_step (int): n-step 學習的步數，計算 n 步累積獎勵。
            alpha (float): 優先級回放的指數，控制優先級程度（0 ≤ α ≤ 1）。
            beta (float): 重要性採樣的初始權重，校正採樣偏差（0 ≤ β ≤ 1）。
            beta_increment (float): 重要性採樣權重的增量。
            expert_prob_start (float): 初始專家策略使用概率。
            expert_prob_end (float): 最終專家策略使用概率。
            expert_prob_decay_steps (int): 專家策略概率衰減的總步數。
        """
        self.state_dim = state_dim  # 狀態空間維度
        self.action_dim = action_dim  # 動作空間維度
        self.device = torch.device(device)  # 計算設備
        self.buffer_size = buffer_size  # 回放緩衝區容量
        self.batch_size = batch_size  # 批量大小
        self.expert_prob = expert_prob_start  # 當前專家策略概率
        self.expert_prob_start = expert_prob_start  # 初始專家概率
        self.expert_prob_end = expert_prob_end  # 最終專家概率
        self.expert_prob_decay_steps = expert_prob_decay_steps  # 專家概率衰減步數
        self.gamma = gamma  # 折扣因子
        self.target_update_freq = target_update_freq  # 目標網絡更新頻率
        self.n_step = n_step  # n-step 學習步數
        self.alpha = alpha  # 優先級回放指數
        self.beta = beta  # 重要性採樣權重
        self.beta_increment = beta_increment  # 重要性採樣權重增量
        self.steps = 0  # 當前訓練步數
        # 初始化主網絡（用於選擇動作和計算 Q 值）
        self.model = DQN(state_dim, action_dim).to(self.device)
        # 初始化目標網絡（用於計算目標 Q 值，穩定訓練）
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # 目標網絡僅用於評估
        # 使用 Adam 優化器，加入 L2 正則化（weight_decay）
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # 使用 SumTree 結構實現優先級經驗回放
        self.memory = SumTree(buffer_size)
        # 用於 n-step 學習的臨時緩衝區
        self.n_step_memory = deque(maxlen=n_step)
        self.max_priority = 1.0  # 初始最大優先級

    def update_expert_prob(self):
        """
        更新專家策略概率，隨訓練步數線性衰減。

        原理：
        - 早期訓練中，代理可能過於依賴隨機探索，導致效率低下。
        - 通過引入專家策略（例如人類玩家或預定義規則），代理可以更快學習有效行為。
        - 隨著訓練進展，專家策略的影響逐漸減少，代理轉向自主學習。
        - 衰減公式：expert_prob = expert_prob_start - (expert_prob_start - expert_prob_end) * (steps / decay_steps)
        """
        decay_progress = min(self.steps / self.expert_prob_decay_steps, 1.0)
        self.expert_prob = self.expert_prob_start - (self.expert_prob_start - self.expert_prob_end) * decay_progress

    def choose_action(self, state):
        """
        根據當前狀態選擇動作（使用 Noisy DQN）。

        原理：
        - Noisy DQN 通過在網絡層（NoisyLinear）中添加隨機噪聲實現探索，無需 ε-貪婪策略。
        - 動作選擇公式：action = argmax(Q(s, a))，其中 Q(s, a) 由主網絡計算。
        - 噪聲使 Q 值具有隨機性，促進探索新動作。

        Args:
            state (array): 當前狀態。

        Returns:
            int: 選擇的動作索引。
        """
        # 將狀態轉為張量並增加批量維度
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()  # 設置為評估模式
        with torch.no_grad():
            q_values = self.model(state)  # 計算 Q 值
        self.model.train()  # 恢復訓練模式
        return q_values.argmax(1).item()  # 返回最大 Q 值的動作索引

    def store_transition(self, state, action, reward, next_state, done):
        """
        將轉換數據存入 n-step 回放緩衝區。

        原理：
        - n-step 學習通過計算未來 n 步的累積獎勵，增強長期獎勵的影響。
        - 累積獎勵公式：R = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ... + γ^(n-1) * r_{t+n-1}
        - 優先級經驗回放根據 TD 誤差分配優先級，優先級越高，採樣概率越大。
        - 轉換數據最終存入 SumTree，優先級初始為最大值（max_priority）。

        Args:
            state (array): 當前狀態。
            action (int): 執行的動作。
            reward (float): 獲得的獎勵。
            next_state (array): 下一個狀態。
            done (bool): 是否終止。
        """
        # 驗證輸入數據的形狀和範圍
        if not isinstance(state, np.ndarray) or state.shape != self.state_dim:
            raise ValueError(f"無效的狀態形狀：預期 {self.state_dim}，得到 {state.shape}")
        if not isinstance(next_state, np.ndarray) or next_state.shape != self.state_dim:
            raise ValueError(f"無效的下一個狀態形狀：預期 {self.state_dim}，得到 {next_state.shape}")
        if not (0 <= action < self.action_dim):
            raise ValueError(f"無效的動作：{action}，預期 0 到 {self.action_dim-1}")

        # 將轉換數據存入 n-step 緩衝區
        state_pacman_x = np.argmax(np.max(state[0], axis=1))
        state_pacman_y = np.argmax(np.max(state[0], axis=0))
        next_state_pacman_x = np.argmax(np.max(next_state[0], axis=1))
        next_state_pacman_y = np.argmax(np.max(next_state[0], axis=0))
        position_change = np.sqrt((state_pacman_x - next_state_pacman_x) ** 2 + 
                                (state_pacman_y - next_state_pacman_y) ** 2)
        if reward >= 0 or position_change > 0.1 or done:
            self.n_step_memory.append((state, action, reward, next_state, done))
        elif position_change == 0 and random.random() > 0.7:
            self.n_step_memory.append((state, action, reward, next_state, done))
        else:
            return
        # self.n_step_memory.append((state, action, reward, next_state, done))
        # 當緩衝區滿或回合結束時，計算 n-step 轉換
        if len(self.n_step_memory) >= self.n_step or done:
            for i in range(len(self.n_step_memory) - (1 if done else 0)):
                total_reward = 0
                final_state = self.n_step_memory[i][0]
                final_action = self.n_step_memory[i][1]
                final_next_state = next_state
                final_done = done
                # 計算 n-step 累積獎勵
                for j in range(i, min(i + self.n_step, len(self.n_step_memory))):
                    r = self.n_step_memory[j][2]
                    total_reward += (self.gamma ** (j - i)) * r
                    if j == len(self.n_step_memory) - 1:
                        final_next_state = self.n_step_memory[j][3]
                        final_done = self.n_step_memory[j][4]
                    if self.n_step_memory[j][4]:
                        final_done = True
                        break
                total_reward = total_reward / 100.0  # 縮放獎勵
                # 創建轉換元組
                transition = Transition(final_state, final_action, total_reward, final_next_state, final_done)
                priority = self.max_priority + 1e-6  # 設置初始優先級
                self.memory.add(priority, transition)  # 存入 SumTree
            if done:
                self.n_step_memory.clear()  # 清空緩衝區
            elif len(self.n_step_memory) >= self.n_step:
                self.n_step_memory.popleft()  # 移除最早的數據

    def sample(self):
        """
        從優先級回放緩衝區採樣一批數據。

        原理：
        - 優先級經驗回放根據轉換的優先級（TD 誤差）採樣，優先級公式：p_i = |TD_error| ^ α
        - 重要性採樣權重用於校正採樣偏差，公式：w_i = (N * P(i)) ^ (-β) / max(w)
        - SumTree 結構高效實現優先級採樣，時間複雜度為 O(log N)。

        Returns:
            tuple: 狀態、動作、獎勵、下一個狀態、是否終止、重要性採樣權重、索引。
        """
        if self.memory.total_priority == 0:
            return None, None, None, None, None, None, None
        indices = []
        batch = []
        weights = []
        # 將總優先級分成 batch_size 份
        segment = self.memory.total_priority / self.batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)  # 更新 β
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)  # 隨機採樣
            idx, priority, data = self.memory.get_leaf(s)
            indices.append(idx)
            batch.append(data)
            prob = priority / self.memory.total_priority  # 採樣概率
            weight = (self.memory.capacity * prob) ** (-self.beta)  # 重要性採樣權重
            weights.append(weight)
        # 解包批量數據
        states, actions, rewards, next_states, dones = zip(*batch)
        # 轉為張量
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

        原理：
        - 模仿學習通過最小化模型預測動作與專家動作之間的交叉熵損失，初始化網絡參數。
        - 損失函數：L = -∑ [y_i * log(π(a|s))，其中：
          - y_i：專家動作（one-hot 編碼）
          - π(a|s)：模型預測的動作概率
        - 預訓練可以讓模型學習專家行為，減少早期隨機探索的低效性。

        Args:
            expert_data (list): 包含 (state, action) 的專家數據。
            pretrain_steps (int): 預訓練步數。
        """
        print(f"Starting pretraining with {len(expert_data)} expert samples for {pretrain_steps} steps...")
        self.model.train()
        for step in range(pretrain_steps):
            # 隨機抽樣
            batch = random.sample(expert_data, min(self.batch_size, len(expert_data)))
            states, actions = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).squeeze()
            # 計算 Q 值並預測動作
            q_values = self.model(states)
            # 計算交叉熵損失
            loss = F.cross_entropy(q_values, actions)
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            if (step + 1) % 100 == 0:
                print(f"Pretrain step {step + 1}/{pretrain_steps}, Loss: {loss.item():.4f}")
        print("Pretraining completed")

    def learn(self, expert_action=False):
        """
        執行一次學習步驟，更新模型參數。

        原理：
        - DQN 通過最小化 TD 誤差更新 Q 值，損失函數：L = E[(y - Q(s, a))^2]
        - 目標 y = r + γ * max Q(s', a')，其中 Q(s', a') 由目標網絡計算。
        - 重要性採樣權重 w_i 校正優先級採樣的偏差。
        - 優先級更新：p_i = |TD_error| + ε，確保新轉換有一定採樣概率。
        - 軟目標更新公式：θ_target ← τ * θ + (1 - τ) * θ_target，穩定學習。

        Args:
            expert_action (bool): 是否為專家動作（若為真，則跳過學習）。

        Returns:
            float: 損失值，若無數據則返回 None。
        """
        if self.memory.total_priority == 0 or len(self.memory.data) < self.batch_size:
            return None
        if expert_action:
            return None
        self.steps += 1
        self.update_expert_prob()
        # 採樣數據
        states, actions, rewards, next_states, dones, weights, indices = self.sample()
        if states is None:  # 檢查採樣是否失敗
            return None
        
        # 初始化 AMP 縮放器
        scaler = GradScaler("cuda")
        self.model.train()  # 確保模型處於訓練模式
        # 使用混合精度訓練
        with autocast("cuda"):
            # 計算當前 Q 值
            q_values = self.model(states).gather(1, actions)
            with torch.no_grad():
                # 使用主模型選擇動作，目標模型計算 Q 值（Double DQN）
                next_actions = self.model(next_states).max(1, keepdim=True)[1]
                next_q_values = self.target_model(next_states).gather(1, next_actions)
                # 計算目標 Q 值
                target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
            # 計算 TD 誤差
            td_errors = (q_values - target_q_values).abs()
            # 計算加權損失
            loss = (td_errors * weights).mean()

        # 優化步驟
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()  # 縮放梯度
        scaler.unscale_(self.optimizer)  # 反縮放以便裁剪
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        scaler.step(self.optimizer)  # 更新參數
        scaler.update()  # 更新縮放器狀態
        
        # 更新優先級
        for i, idx in enumerate(indices):
            priority = td_errors[i].detach().cpu().numpy()[0] + 1e-6
            self.memory.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
        # 定期軟更新目標網絡
        if self.steps % self.target_update_freq == 0:
            tau = 0.005  # 軟更新係數
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        return loss.item()

    def save(self, model_path, memory_path):
        """
        保存模型和記憶數據。

        Args:
            model_path (str): 模型保存路徑。
            memory_path (str): 記憶數據保存路徑。
        """
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(list(self.memory.data), f)
        print(f"Saved model to {model_path} and memory to {memory_path}")

    def load(self, model_path, memory_path=None):
        """
        載入模型和記憶數據。

        Args:
            model_path (str): 模型文件路徑。
            memory_path (str, optional): 記憶數據文件路徑。
        """
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