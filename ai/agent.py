import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
import os
from collections import deque, namedtuple
from dqn import DQN, NoisyLinear
from sumtree import SumTree

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=500000, batch_size=32, 
                 lr=2.5e-4, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay_steps=1000000, 
                 gamma=0.99, target_update_freq=200, n_step=3, alpha=0.6, beta=0.4, beta_increment=0.001):
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
        self.n_step = n_step
        self.alpha = alpha  # Prioritized Replay 優先級指數
        self.beta = beta    # IS 權重的重要性採樣
        self.beta_increment = beta_increment
        self.steps = 0

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = SumTree(buffer_size)
        self.n_step_memory = deque(maxlen=n_step)
        self.max_priority = 1.0

    def update_epsilon(self):
        decay_progress = min(self.steps / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()
        return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.n_step_memory.append((state, action, reward, next_state, done))
        if len(self.n_step_memory) == self.n_step or done:
            total_reward = 0
            final_state = next_state
            final_done = done
            for i, (s, a, r, ns, d) in enumerate(reversed(self.n_step_memory)):
                total_reward = r + self.gamma * total_reward
                if i == 0:
                    final_state = s
                if d:
                    break
            transition = Transition(final_state, a, total_reward, next_state, done)
            priority = self.max_priority if not self.memory.total_priority else self.max_priority
            self.memory.add(priority, transition)
            self.n_step_memory.clear()

    def sample(self):
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

    def learn(self):
        if self.memory.total_priority == 0 or len(self.memory.data) < self.batch_size:
            return None

        self.steps += 1
        self.update_epsilon()
        self.model.reset_noise()  # Noisy Net 重置噪聲
        states, actions, rewards, next_states, dones, weights, indices = self.sample()
        # print(f"States shape: {states.shape}")
        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.model(next_states).max(1, keepdim=True)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = (q_values - target_q_values).abs()
        loss = (td_errors * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新優先級
        for i, idx in enumerate(indices):
            priority = td_errors[i].detach().cpu().numpy()[0] + 1e-6  # 避免零優先級
            self.memory.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save(self, model_path, memory_path):
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(list(self.memory.data), f)  # 儲存 data 陣列
        print(f"Saved model to {model_path} and memory to {memory_path}")

    def load(self, model_path, memory_path=None):
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