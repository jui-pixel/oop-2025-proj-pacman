"""
定義 DQN 代理，負責管理記憶緩衝區、動作選擇和模型訓練。
支援優先級經驗回放 (PER)、模型保存和載入，實現深度強化學習的核心邏輯。
此代理基於 Double DQN，並使用優先級經驗回放來優化學習效率。
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
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=50000, batch_size=128, lr=5e-4, epsilon=0.9):
        """
        初始化 DQN 代理，設置主模型、目標模型、記憶緩衝區和訓練參數。
        此方法創建一個深度 Q 網路代理，用於訓練 Pac-Man 遊戲中的強化學習策略。

        Args:
            state_dim (Tuple[int, int, int]): 狀態維度 (高度, 寬度, 通道數)，例如 (31, 28, 6)，表示環境狀態的形狀。
            action_dim (int): 動作數量，例如 4（上、下、左、右），表示可用的動作空間。
            device (str): 計算設備，"cuda" 或 "cpu"，預設為 "cpu"，決定模型運行在哪個設備上。
            buffer_size (int): 記憶緩衝區的最大容量，預設為 50000，限制經驗數據的儲存數量。
            batch_size (int): 每次訓練的批次大小，預設為 128，控制訓練中一次處理的樣本數。
            lr (float): 學習率，預設為 5e-4，控制模型參數更新的步長。
            epsilon (float): 初始探索率，預設為 0.9，用於 ε-貪婪策略的探索行為。
        """
        self.state_dim = state_dim  # 保存狀態維度，用於後續模型初始化
        self.action_dim = action_dim  # 保存動作數量
        self.device = device  # 保存計算設備
        self.memory = deque(maxlen=buffer_size)  # 初始化雙端隊列作為經驗回放緩衝區，限制最大長度
        self.priorities = deque(maxlen=buffer_size)  # 儲存優先級值，用於優先級經驗回放
        self.batch_size = batch_size  # 設置批次大小
        self.gamma = 0.995  # 折扣因子，控制未來獎勵的重要性，接近 1 表示重視長期獎勵
        self.epsilon = epsilon  # 當前探索率，初始值由參數指定
        self.epsilon_min = 0.05  # 探索率的最小值，避免完全停止探索
        self.model = DQN(state_dim, action_dim).to(device)  # 初始化主模型並移動到指定設備
        self.target_model = DQN(state_dim, action_dim).to(device)  # 初始化目標模型並移動到指定設備
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # 使用 Adam 優化器，設置學習率
        self.steps = 0  # 跟踪訓練步數，用於控制目標網絡更新
        self.target_update_freq = 300  # 目標網絡更新頻率，每 300 步更新一次
        self.alpha = 0.6  # 優先級經驗回放的 alpha 參數，控制優先級的影響程度
        self.beta = 0.5  # 重要性採樣的初始 beta 參數，隨時間增加以平衡偏差
        self.beta_increment = 0.002  # beta 每次增加的步長
        self.last_action = None  # 記錄上一次執行的動作，幫助調整動作選擇權重
        self.last_position = None  # 記錄 Pac-Man 上一次的位置，檢測停滯
        self.stuck_counter = 0  # 計數 Pac-Man 連續停滯的次數
        self.action_cooldown = 0  # 動作冷卻計數器，限制連續相同動作
        self.cooldown_steps = 0  # 冷卻步數，當前未設置具體值
        self.recent_rewards = deque(maxlen=100)  # 儲存最近 100 個獎勵，用於分析表現
        self.update_target_model()  # 初始化時同步主模型和目標模型

    def update_target_model(self):
        """
        將主模型的權重復製到目標模型，確保兩者一致。
        這是 Double DQN 中的關鍵步驟，穩定目標 Q 值估計。
        """
        self.target_model.load_state_dict(self.model.state_dict())  # 直接複製主模型參數

    def _is_valid_action(self, state, action):
        """
        檢查動作是否有效（是否會撞牆）。
        根據狀態中的牆壁通道（通道 5）判斷移動是否合法。

        Args:
            state (numpy.ndarray): 當前狀態，形狀為 (height, width, channels)。
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。

        Returns:
            bool: 動作是否有效，若新位置無牆壁則返回 True。
        """
        # 找到 Pac-Man 的當前位置（通道 0 為 1.0 的點）
        pacman_x, pacman_y = np.where(state[:, :, 0] == 1.0)
        if len(pacman_x) == 0:  # 如果未找到 Pac-Man，認為動作有效（應改進邏輯）
            return True
        pacman_x, pacman_y = pacman_x[0], pacman_y[0]  # 取第一個匹配點

        # 根據動作計算移動方向
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x, new_y = pacman_x + dx, pacman_y + dy  # 計算新位置

        # 檢查新位置是否在範圍內且無牆壁（通道 5 為 0 表示無障礙）
        if 0 <= new_x < state.shape[0] and 0 <= new_y < state.shape[1]:
            return state[new_x, new_y, 5] != 1.0
        return False  # 超出邊界視為無效

    def _check_stuck(self, state):
        """
        檢查 Pac-Man 是否停滯（位置未改變）。
        用於檢測代理是否陷入死循環，觸發停滯時可能需要額外懲罰。

        Args:
            state (numpy.ndarray): 當前狀態。

        Returns:
            bool: 是否停滯，當連續停滯次數達到 2 時返回 True。
        """
        # 找到 Pac-Man 的當前位置
        pacman_x, pacman_y = np.where(state[:, :, 0] == 1.0)
        if len(pacman_x) == 0:  # 如果未找到 Pac-Man，返回 False
            return False
        current_position = (pacman_x[0], pacman_y[0])  # 取第一個匹配點

        # 比較當前與上次位置，更新停滯計數
        if self.last_position is not None and current_position == self.last_position:
            self.stuck_counter += 1  # 增加停滯計數
        else:
            self.stuck_counter = 0  # 重置計數
        self.last_position = current_position  # 更新上次位置
        return self.stuck_counter >= 2  # 閾值為 2，可根據需要調整

    def get_action(self, state):
        """
        使用基於權重的 ε-貪婪策略選擇動作，根據 last_action 調整權重。
        結合探索（隨機選擇）和利用（基於 Q 值選擇）策略。

        Args:
            state (numpy.ndarray): 當前狀態，形狀為 (height, width, channels)。

        Returns:
            int: 選擇的動作索引（0: 上, 1: 下, 2: 左, 3: 右）。
        """
        if self.action_cooldown > 0:
            # 如果處於冷卻狀態，減少冷卻計數並返回上一次動作
            self.action_cooldown -= 1
            return self.last_action if self.last_action is not None else random.randrange(self.action_dim)

        # 過濾出有效的動作（不會撞牆）
        valid_actions = [a for a in range(self.action_dim) if self._is_valid_action(state, a)]
        if not valid_actions:
            # 如果無效動作，觸發冷卻並隨機選擇
            self.action_cooldown = self.cooldown_steps
            return random.randrange(self.action_dim)

        is_stuck = self._check_stuck(state)  # 檢查是否停滯

        if random.random() < self.epsilon or is_stuck:
            # 探索階段：根據 epsilon 或停滯觸發隨機選擇
            # 根據上一次動作調整權重，鼓勵改變方向
            base_weights = [1.0] * self.action_dim
            if self.last_action is not None:
                if self.last_action == 0:  # 上一步是上
                    weights = [1.7, 0.3, 1.0, 1.0]  # 提高上方向權重，降低反方向
                elif self.last_action == 1:  # 上一步是下
                    weights = [0.3, 1.7, 1.0, 1.0]
                elif self.last_action == 2:  # 上一步是左
                    weights = [1.0, 1.0, 1.7, 0.3]
                elif self.last_action == 3:  # 上一步是右
                    weights = [1.0, 1.0, 0.3, 1.7]
            else:
                weights = base_weights

            # 將權重應用到有效動作
            action_weights = [weights[a] for a in valid_actions]
            if sum(action_weights) == 0:  # 防止除以零
                action_weights = [1.0] * len(valid_actions)
            else:
                action_weights = [w / sum(action_weights) for w in action_weights]  # 正規化

            # 根據權重隨機選擇動作
            action = random.choices(valid_actions, weights=action_weights, k=1)[0]
        else:
            # 利用階段：根據 Q 值選擇最佳動作
            with torch.no_grad():  # 禁用梯度計算以節省記憶體
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)  # 獲取 Q 值
                q_values_np = q_values.cpu().numpy()[0]  # 轉為 NumPy 陣列
                q_values_valid = [(q_values_np[a], a) for a in valid_actions]  # 過濾有效動作的 Q 值
                if not q_values_valid:
                    action = random.choice(valid_actions)  # 如果無效 Q 值，隨機選擇
                else:
                    action = max(q_values_valid, key=lambda x: x[0])[1]  # 選擇 Q 值最大的動作

        self.last_action = action  # 更新上一次動作
        self.action_cooldown = self.cooldown_steps  # 重置冷卻計數器
        return action

    def train(self):
        """
        從記憶緩衝區採樣並訓練主模型，使用優先級經驗回放和重要性採樣，實現 Double DQN。
        這是代理的核心訓練邏輯，基於 TD 誤差更新 Q 值。

        Returns:
            float: 當前批次的損失值，若記憶不足則返回 0。
        """
        if len(self.memory) < self.batch_size:
            # 如果記憶緩衝區數據不足，返回 0 表示不訓練
            return 0.0

        # 計算優先級並正規化為概率
        priorities = np.array(self.priorities) ** self.alpha  # 應用 alpha 控制優先級影響
        probabilities = priorities / priorities.sum()  # 正規化為概率分佈
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)  # 根據概率採樣
        batch = [self.memory[i] for i in indices]  # 獲取採樣的經驗

        # 解包批次數據並轉為張量
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)  # 調整通道維度
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # 動作索引
        rewards = torch.FloatTensor(rewards).to(self.device)  # 獎勵
        next_states = torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(self.device)  # 下一個狀態
        dones = torch.FloatTensor(dones).to(self.device)  # 結束標誌

        # 計算當前 Q 值
        q_values = self.model(states).gather(1, actions).squeeze(1)  # 從 Q 值中提取選定動作的值

        # Double DQN：使用主模型選擇動作，目標模型評估 Q 值
        with torch.no_grad():  # 禁用梯度計算
            next_actions = self.model(next_states).argmax(1, keepdim=True)  # 選擇最佳動作
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)  # 評估 Q 值
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values  # 計算目標 Q 值

        # 更新優先級
        td_errors = (q_values - target_q_values).abs().cpu().detach().numpy()  # 計算 TD 誤差
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error + 1e-5  # 添加小值避免零優先級

        # 計算重要性採樣權重
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)  # 根據 beta 調整權重
        weights /= weights.max()  # 正規化
        weights = torch.FloatTensor(weights).to(self.device)  # 轉為張量

        # 計算損失並更新模型
        loss = (weights * nn.MSELoss(reduction='none')(q_values, target_q_values)).mean()  # 加權 MSE 損失
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向傳播
        self.optimizer.step()  # 更新參數

        self.steps += 1  # 增加步數計數
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()  # 定期更新目標模型
        self.beta = min(1.0, self.beta + self.beta_increment)  # 逐漸增加 beta

        return loss.item()  # 返回當前損失值

    def remember(self, state, action, reward, next_state, done):
        """
        將經驗儲存到記憶緩衝區，並為新經驗設置最高優先級。
        這是經驗回放的數據收集過程，支援優先級更新。

        Args:
            state (numpy.ndarray): 當前狀態。
            action (int): 執行的動作。
            reward (float): 獲得的獎勵。
            next_state (numpy.ndarray): 下一個狀態。
            done (bool): 是否結束。
        """
        max_priority = max(self.priorities) if self.priorities else 1.0  # 獲取最高優先級，初始為 1.0
        self.memory.append((state, action, reward, next_state, done))  # 儲存經驗
        self.priorities.append(max_priority)  # 為新經驗設置初始優先級

    def save(self, path, memory_path=None):
        """
        保存模型權重、優化器狀態、探索率和記憶緩衝區。
        允許將訓練狀態持久化，支援後續恢復。

        Args:
            path (str): 模型檔案路徑（.pth）。
            memory_path (str, optional): 記憶緩衝區檔案路徑（.pkl）。
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  # 保存主模型參數
            'optimizer_state_dict': self.optimizer.state_dict(),  # 保存優化器狀態
            'epsilon': self.epsilon,  # 保存當前探索率
            'steps': self.steps  # 保存步數
        }
        torch.save(checkpoint, path)  # 保存模型檢查點
        if memory_path:
            with open(memory_path, 'wb') as f:
                pickle.dump(list(self.memory), f)  # 保存記憶緩衝區

    def load(self, path, memory_path=None):
        """
        載入模型權重、優化器狀態、探索率和記憶緩衝區。
        支援從先前訓練點恢復代理狀態。

        Args:
            path (str): 模型檔案路徑（.pth）。
            memory_path (str, optional): 記憶緩衝區檔案路徑（.pkl）。
        """
        checkpoint = torch.load(path)  # 載入模型檢查點
        self.model.load_state_dict(checkpoint['model_state_dict'])  # 恢復主模型參數
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 恢復優化器狀態
        self.epsilon = checkpoint['epsilon']  # 恢復探索率
        self.steps = checkpoint['steps']  # 恢復步數
        self.update_target_model()  # 同步目標模型
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory = pickle.load(f)  # 載入記憶緩衝區
                self.memory = deque(memory, maxlen=self.memory.maxlen)  # 恢復記憶緩衝區