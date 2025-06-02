import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
import os
from ai.dqn import DuelingDQN
from ai.sumtree import SumTree

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=50000, batch_size=64, lr=2.5e-4,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50000,
                 gamma=0.99, n_step=1, alpha=0.6, beta=0.4, beta_annealing_steps=100000,
                 priority_epsilon=1e-5, target_update_freq=200):
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
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.priority_epsilon = priority_epsilon
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = SumTree(buffer_size)

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
        max_priority = self.memory.total_priority ** self.alpha if self.memory.total_priority > 0 else 1.0
        self.memory.add(max_priority, (state, action, reward, next_state, done))

    def sample(self):
        batch = []
        indices = []
        priorities = []
        segment = self.memory.total_priority / self.batch_size

        self.beta = min(1.0, self.beta + (1.0 - self.beta) / self.beta_annealing_steps)
        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            idx, priority, data = self.memory.get_leaf(v)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = np.array(priorities) / self.memory.total_priority
        weights = (self.memory.capacity * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones, indices, weights

    def learn(self):
        if self.memory.data_pointer < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.sample()
        self.steps += 1
        self.update_epsilon()

        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN: select actions using model, evaluate using target_model
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        loss = (weights * F.mse_loss(q_values, target_q_values, reduction='none')).mean()

        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        for idx, error in zip(indices, td_errors.flatten()):
            priority = (error + self.priority_epsilon) ** self.alpha
            self.memory.update(idx, priority)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save(self, model_path, memory_path):
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, model_path, memory_path=None):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)