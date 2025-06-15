# ai/agent.py
# 導入 PyTorch 相關模組，用於建立和訓練神經網絡
import torch
# 導入 PyTorch 的神經網絡模組，提供層類型和損失函數
import torch.nn as nn
# 導入優化器模組，用於更新模型參數
import torch.optim as optim
# 導入 PyTorch 的函數操作模組，例如激活函數
import torch.nn.functional as F
# 導入 random 模組，用於隨機選擇動作或生成隨機數
import random
# 導入 os 模組，用於處理檔案路徑
import os
# 導入 numpy，用於數值計算和陣列操作
import numpy as np
# 導入 pickle，用於序列化和反序列化記憶數據
import pickle
# 導入 deque 和 namedtuple，用於實現回放緩衝區和轉換數據結構
from collections import deque, namedtuple
# 從 ai.dqn 匯入 DQN 和 NoisyLinear 類
from ai.dqn import DQN, NoisyLinear
# 從 ai.sumtree 匯入 SumTree 類，用於優先級回放緩衝區
from ai.sumtree import SumTree
# 導入自動混合精度模組，提升訓練效率
from torch.amp import autocast, GradScaler
# 從 config 檔案匯入常數參數，例如緩衝區大小、學習率等
from config import *
# 定義轉換數據結構，包含狀態、動作、獎勵、下一個狀態和結束標誌
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, 
                 lr=LEARNING_RATE, gamma=GAMMA, target_update_freq=TARGET_UPDATE_FREQ, n_step=N_STEP, alpha=ALPHA, beta=BETA, 
                 beta_increment=BETA_INCREMENT, expert_prob_start=EXPERT_PROB_START, expert_prob_end=EXPERT_PROB_END, 
                 expert_prob_decay_steps=EXPERT_PROB_DECAY_STEPS, sigma=SIGMA):
        """
        初始化 DQN 代理。

        這個類實現了 DQN（Deep Q-Network）代理，負責與環境交互、選擇動作、儲存經驗並進行學習。

        Args:
            state_dim (tuple): 狀態維度，形狀為 (通道數, 高度, 寬度)。
            action_dim (int): 動作數量（動作空間大小）。
            device (str): 訓練設備（"cpu" 或 "cuda"）。
            buffer_size (int): 回放緩衝區大小。
            batch_size (int): 批量大小，用於學習。
            lr (float): 學習率。
            gamma (float): 折扣因子，用於計算未來獎勵。
            target_update_freq (int): 目標網絡更新頻率。
            n_step (int): N 步回報的步數。
            alpha (float): 優先級回放的 alpha 參數，控制優先級權重。
            beta (float): 優先級回放的 beta 參數，控制重要性採樣。
            beta_increment (float): beta 每次增加的值。
            expert_prob_start (float): 專家動作概率的起始值。
            expert_prob_end (float): 專家動作概率的終止值。
            expert_prob_decay_steps (int): 專家概率衰減的步數。
            sigma (float): Noisy DQN 的噪聲因子。
        """
        # 儲存狀態和動作維度
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 設定訓練設備（GPU 或 CPU）
        self.device = torch.device(device)
        # 儲存緩衝區和批量大小
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        # 初始化專家動作概率
        self.expert_prob = expert_prob_start
        self.expert_prob_start = expert_prob_start
        self.expert_prob_end = expert_prob_end
        self.expert_prob_decay_steps = expert_prob_decay_steps
        # 儲存折扣因子
        self.gamma = gamma
        # 儲存目標網絡更新頻率
        self.target_update_freq = target_update_freq
        # 儲存 N 步回報的步數
        self.n_step = n_step
        # 儲存優先級回放參數
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        # 初始化步數計數器
        self.steps = 0
        # 初始化主模型（DQN）
        self.model = DQN(state_dim, action_dim, 3, sigma).to(self.device)
        # 初始化目標模型，與主模型結構相同
        self.target_model = DQN(state_dim, action_dim, 3, sigma).to(self.device)
        # 將主模型的參數複製到目標模型
        self.target_model.load_state_dict(self.model.state_dict())
        # 設置目標模型為評估模式（不進行梯度更新）
        self.target_model.eval()
        # 初始化 Adam 優化器，加入權重衰減以防止過擬合
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # 初始化優先級回放緩衝區（使用 SumTree 結構）
        self.memory = SumTree(buffer_size)
        # 初始化 N 步回放緩衝區
        self.n_step_memory = deque(maxlen=n_step)
        # 初始化最大優先級，用於優先級回放
        self.max_priority = 1.0

    def update_expert_prob(self):
        """
        更新專家策略概率。

        隨著訓練步數增加，專家動作的概率會從起始值逐漸衰減到終止值。
        """
        # 計算衰減進度，限制在 [0, 1]
        decay_progress = min(self.steps / self.expert_prob_decay_steps, 1.0)
        # 線性衰減專家概率
        self.expert_prob = self.expert_prob_start - (self.expert_prob_start - self.expert_prob_end) * decay_progress

    def choose_action(self, state):
        """
        根據當前狀態選擇動作（使用 Noisy DQN）。

        Args:
            state (np.ndarray): 當前狀態，形狀為 (通道數, 高度, 寬度)。

        Returns:
            int: 選擇的動作編號。
        """
        # 將狀態轉換為 PyTorch 張量並添加批次維度
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # 設置模型為評估模式（不計算梯度）
        self.model.eval()
        # 不追蹤梯度以節省記憶體
        with torch.no_grad():
            # 計算 Q 值
            q_values, _ = self.model(state)
        # 恢復訓練模式
        self.model.train()
        # 選擇 Q 值最大的動作
        return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        儲存轉換數據至 N 步回放緩衝區。

        這個方法將狀態、動作、獎勵等數據儲存到 N 步緩衝區，並在滿足條件時計算 N 步回報並存入優先級回放緩衝區。

        Args:
            state (np.ndarray): 當前狀態。
            action (int): 執行的動作。
            reward (float): 獲得的獎勵。
            next_state (np.ndarray): 下一個狀態。
            done (bool): 是否結束。
        """
        # 驗證狀態形狀
        if not isinstance(state, np.ndarray) or state.shape != self.state_dim:
            raise ValueError(f"無效的狀態形狀：預期 {self.state_dim}，得到 {state.shape}")
        if not isinstance(next_state, np.ndarray) or next_state.shape != self.state_dim:
            raise ValueError(f"無效的下一個狀態形狀：預期 {self.state_dim}，得到 {next_state.shape}")
        # 驗證動作有效性
        if not (0 <= action < self.action_dim):
            raise ValueError(f"無效的動作：{action}")
        # 將轉換數據添加到 N 步緩衝區
        self.n_step_memory.append((state, action, reward, next_state, done))
        # 如果緩衝區已滿或回合結束，則計算 N 步回報
        if len(self.n_step_memory) >= self.n_step or done:
            for i in range(len(self.n_step_memory) - (1 if done else 0)):
                total_reward = 0
                final_state = self.n_step_memory[i][0]
                final_action = self.n_step_memory[i][1]
                final_next_state = next_state
                final_done = done
                # 計算 N 步累計獎勵
                for j in range(i, min(i + self.n_step, len(self.n_step_memory))):
                    r = self.n_step_memory[j][2]
                    total_reward += (self.gamma ** (j - i)) * r
                    if j == len(self.n_step_memory) - 1:
                        final_next_state = self.n_step_memory[j][3]
                        final_done = self.n_step_memory[j][4]
                    if self.n_step_memory[j][4]:
                        final_done = True
                        break
                # 縮放獎勵以穩定訓練
                total_reward = total_reward / 100.0
                # 創建轉換數據
                transition = Transition(final_state, final_action, total_reward, final_next_state, final_done)
                # 設定優先級
                priority = self.max_priority + 1e-6
                # 將轉換數據添加到優先級回放緩衝區
                self.memory.add(priority, transition)
            # 如果回合結束，清空 N 步緩衝區
            if done:
                self.n_step_memory.clear()
            # 否則移除最早的數據
            elif len(self.n_step_memory) >= self.n_step:
                self.n_step_memory.popleft()

    def sample(self):
        """
        從優先級回放緩衝區採樣一批數據。

        使用 SumTree 結構根據優先級進行採樣，並計算重要性採樣權重。

        Returns:
            tuple: (狀態, 動作, 獎勵, 下一個狀態, 結束標誌, 權重, 索引)。
        """
        # 如果緩衝區為空，返回 None
        if self.memory.total_priority == 0:
            return None, None, None, None, None, None, None
        indices = []
        batch = []
        weights = []
        # 將總優先級分成 batch_size 份
        segment = self.memory.total_priority / self.batch_size
        # 更新 beta 值，控制重要性採樣
        self.beta = min(1.0, self.beta + self.beta_increment)
        # 對每個批次進行採樣
        for i in range(self.batch_size):
            # 在當前分段內隨機選擇一個優先級
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            # 從 SumTree 獲取數據
            idx, priority, data = self.memory.get_leaf(s)
            indices.append(idx)
            batch.append(data)
            # 計算重要性採樣權重
            prob = priority / self.memory.total_priority
            weight = (self.memory.capacity * prob) ** (-self.beta)
            weights.append(weight)
        # 將批次數據解包
        states, actions, rewards, next_states, dones = zip(*batch)
        # 轉換為 PyTorch 張量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        # 正規化權重
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1) / max(weights)
        return states, actions, rewards, next_states, dones, weights, indices

    def pretrain(self, expert_data, pretrain_steps=1000):
        """
        使用專家數據進行模仿學習預訓練。

        Args:
            expert_data (list): 包含 (狀態, 動作) 元組的專家數據列表。
            pretrain_steps (int): 預訓練步數，預設為 1000。
        """
        print(f"開始預訓練，專家數據量：{len(expert_data)}，預訓練步數：{pretrain_steps}")
        # 設置模型為訓練模式
        self.model.train()
        # 初始化梯度縮放器，用於混合精度訓練
        scaler = GradScaler("cuda" if self.device.type == "cuda" else "cpu")
        # 進行預訓練循環
        for step in range(pretrain_steps):
            # 隨機採樣一批專家數據
            batch = random.sample(expert_data, min(self.batch_size, len(expert_data)))
            states, actions = zip(*batch)
            # 轉換為 PyTorch 張量
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).squeeze()
            # 使用自動混合精度進行前向傳播
            if self.device.type == "cuda":
                with autocast("cuda"):
                    q_values, _ = self.model(states)
                    # 使用交叉熵損失進行模仿學習
                    loss = F.cross_entropy(q_values, actions)
            else:
                q_values, _ = self.model(states)
                loss = F.cross_entropy(q_values, actions)
            # 清空梯度
            self.optimizer.zero_grad()
            # 執行反向傳播和優化
            if self.device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                # 裁剪梯度，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
            # 每 100 步印出損失值
            if (step + 1) % 100 == 0:
                print(f"預訓練步數 {step + 1}/{pretrain_steps}，損失：{loss.item():.2f}")
        print("預訓練完成")

    def learn(self, expert_action=False):
        """
        執行一次 DQN 學習步驟。

        使用優先級回放和 N 步回報計算損失，並更新模型參數。

        Args:
            expert_action (bool): 是否為專家動作（專家動作不進行學習）。

        Returns:
            float or None: 損失值，若無法學習則返回 None。
        """
        # 如果緩衝區數據不足或為專家動作，則跳過學習
        if self.memory.total_priority == 0 or len(self.memory.data) < self.batch_size:
            return None
        if expert_action:
            return None
        
        # 增加步數並更新專家概率
        self.steps = self.steps + 1
        self.update_expert_prob()
        # 從回放緩衝區採樣
        states, actions, rewards, next_states, dones, weights, indices = self.sample()
        if states is None:
            return None
        # 初始化梯度縮放器
        scaler = GradScaler("cuda")
        # 設置模型為訓練模式
        self.model.train()
        # 使用自動混合精度進行前向傳播
        with autocast("cuda"):
            # 計算當前狀態的 Q 值
            q_values, _ = self.model(states)
            q_values = q_values.gather(1, actions)
            # 使用目標模型計算下一個狀態的 Q 值
            with torch.no_grad():
                # Double DQN：用主模型選擇動作，目標模型計算 Q 值
                next_actions = self.model(next_states)[0].max(1, keepdim=True)[1]
                next_q_values = self.target_model(next_states)[0].gather(1, next_actions)
                # 計算目標 Q 值（N 步回報）
                target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
            # 計算 TD 誤差
            td_errors = (q_values - target_q_values).abs()
            # 計算加權平均損失
            loss = (td_errors * weights).mean()
        # 清空梯度
        self.optimizer.zero_grad()
        # 執行反向傳播和優化
        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        # 裁剪梯度，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        scaler.step(self.optimizer)
        scaler.update()
        # 更新優先級
        for i, idx in enumerate(indices):
            priority = td_errors[i].detach().cpu().numpy()[0] + 1e-6
            self.memory.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
        # 定期更新目標網絡（使用軟更新）
        if self.steps % self.target_update_freq == 0:
            tau = 0.01
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        # 返回損失值
        return loss.item()

    def save(self, model_path, memory_path):
        """
        儲存模型和回放緩衝區數據。

        Args:
            model_path (str): 模型儲存路徑。
            memory_path (str): 回放緩衝區儲存路徑。
        """
        # 儲存主模型參數
        torch.save(self.model.state_dict(), model_path)
        # 儲存回放緩衝區數據
        with open(memory_path, 'wb') as f:
            pickle.dump(list(self.memory.data), f)
        print(f"保存模型到 {model_path}，記憶到 {memory_path}")

    def load(self, model_path, memory_path=None):
        """
        載入模型和回放緩衝區數據。

        Args:
            model_path (str): 模型載入路徑。
            memory_path (str, optional): 回放緩衝區載入路徑。
        """
        # 載入主模型參數
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # 將主模型參數複製到目標模型
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"已從 {model_path} 載入模型")