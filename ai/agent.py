# ai/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import numpy as np
import pickle
from collections import deque, namedtuple
from ai.dqn import DQN, NoisyLinear
from ai.sumtree import SumTree
from torch.amp import autocast, GradScaler
from config import *
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, 
                 lr=LEARNING_RATE, gamma=GAMMA, target_update_freq=TARGET_UPDATE_FREQ, n_step=N_STEP, alpha=ALPHA, beta=BETA, 
                 beta_increment=BETA_INCREMENT, expert_prob_start=EXPERT_PROB_START, expert_prob_end=EXPERT_PROB_END, 
                 expert_prob_decay_steps=EXPERT_PROB_DECAY_STEPS, sigma = SIGMA):
        """
        初始化 DQN 代理。
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
        self.model = DQN(state_dim, action_dim, 3, sigma).to(self.device)
        self.target_model = DQN(state_dim, action_dim, 3, sigma).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = SumTree(buffer_size)
        self.n_step_memory = deque(maxlen=n_step)
        self.max_priority = 1.0

    def update_expert_prob(self):
        """
        更新專家策略概率。
        """
        decay_progress = min(self.steps / self.expert_prob_decay_steps, 1.0)
        self.expert_prob = self.expert_prob_start - (self.expert_prob_start - self.expert_prob_end) * decay_progress

    def choose_action(self, state):
        """
        選擇動作（Noisy DQN）。
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values, _ = self.model(state)
        self.model.train()
        return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        儲存轉換數據至 n-step 緩衝區。
        """
        if not isinstance(state, np.ndarray) or state.shape != self.state_dim:
            raise ValueError(f"無效的狀態形狀：預期 {self.state_dim}，得到 {state.shape}")
        if not isinstance(next_state, np.ndarray) or next_state.shape != self.state_dim:
            raise ValueError(f"無效的下一個狀態形狀：預期 {self.state_dim}，得到 {next_state.shape}")
        if not (0 <= action < self.action_dim):
            raise ValueError(f"無效的動作：{action}")
        # state_pacman_x = np.argmax(np.max(state[0], axis=1))
        # state_pacman_y = np.argmax(np.max(state[0], axis=0))
        # next_state_pacman_x = np.argmax(np.max(next_state[0], axis=1))
        # next_state_pacman_y = np.argmax(np.max(next_state[0], axis=0))
        # position_change = np.sqrt((state_pacman_x - next_state_pacman_x) ** 2 + 
        #                           (state_pacman_y - next_state_pacman_y) ** 2)
        # if reward >= 0 or position_change != 0 or done:
        #     self.n_step_memory.append((state, action, reward, next_state, done))
        # elif position_change == 0 and random.random() > 0.5:
        #     self.n_step_memory.append((state, action, reward, next_state, done))
        # else:
        #     return
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
        從優先級回放緩衝區採樣。
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
        模仿學習預訓練。
        """
        print(f"開始預訓練，專家數據量：{len(expert_data)}，預訓練步數：{pretrain_steps}")
        self.model.train()
        scaler = GradScaler("cuda" if self.device.type == "cuda" else "cpu")
        for step in range(pretrain_steps):
            batch = random.sample(expert_data, min(self.batch_size, len(expert_data)))
            states, actions = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).squeeze()
            if self.device.type == "cuda":
                with autocast("cuda"):
                    q_values, _ = self.model(states)
                    loss = F.cross_entropy(q_values, actions)
            else:
                q_values, _ = self.model(states)
                loss = F.cross_entropy(q_values, actions)
            self.optimizer.zero_grad()
            if self.device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
            if (step + 1) % 100 == 0:
                print(f"預訓練步數 {step + 1}/{pretrain_steps}，損失：{loss.item():.2f}")
        print("預訓練完成")

    def learn(self, expert_action=False):
        """
        執行一次學習步驟。
        """
        if self.memory.total_priority == 0 or len(self.memory.data) < self.batch_size:
            # print(f"跳過學習：緩衝區空或數據不足，步數 {self.steps}")
            return None
        if expert_action:
            # print(f"跳過學習：專家行動，步數 {self.steps}")
            return None
        
        self.steps = self.steps + 1
        self.update_expert_prob()
        states, actions, rewards, next_states, dones, weights, indices = self.sample()
        if states is None:
            return None
        scaler = GradScaler("cuda")
        self.model.train()
        with autocast("cuda"):
            q_values, _ = self.model(states)
            q_values = q_values.gather(1, actions)
            with torch.no_grad():
                next_actions = self.model(next_states)[0].max(1, keepdim=True)[1]
                next_q_values = self.target_model(next_states)[0].gather(1, next_actions)
                target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
            td_errors = (q_values - target_q_values).abs()
            loss = (td_errors * weights).mean()
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        scaler.step(self.optimizer)
        scaler.update()
        for i, idx in enumerate(indices):
            priority = td_errors[i].detach().cpu().numpy()[0] + 1e-6
            self.memory.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
        if self.steps % self.target_update_freq == 0:
            tau = 0.001
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        # print(f"學習完成，損失 {loss.item():.4f}，步數 {self.steps}")
        return loss.item()

    def save(self, model_path, memory_path):
        """
        保存儲存模型和記憶數據。
        """
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(list(self.memory.data), f)
        print(f"保存模型到 {model_path}，記憶到 {memory_path}")

    def load(self, model_path, memory_path=None):
        """
        Load model and memory data.
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"已從 {model_path} 載入模型")