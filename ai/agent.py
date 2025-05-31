# ai/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
from ai.dqn import DuelingDQN
from ai.sumtree import SumTree

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", buffer_size=100000, batch_size=128, lr=1e-4, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=100000, epsilon_warmup_steps=1000,
                 gamma=0.99, n_step=1, alpha=0.6, beta=0.4, beta_annealing_steps=200000, 
                 priority_epsilon=1e-5, target_update_freq=100):
        """
        Initialize the DQN agent with enhanced epsilon-random exploration.

        Args:
            state_dim (Tuple[int, int, int]): State dimension (C, H, W), e.g., (6, 31, 28).
            action_dim (int): Number of actions (4).
            epsilon_start (float): Initial epsilon value.
            epsilon_end (float): Minimum epsilon value.
            epsilon_decay_steps (int): Steps to decay epsilon linearly.
            epsilon_warmup_steps (int): Steps for epsilon warmup.
            ... (other args unchanged)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_warmup_steps = epsilon_warmup_steps
        self.epsilon = 0.1  # Start low, warmup to epsilon_start
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
        """
        Update epsilon with warmup and linear decay.
        """
        if self.steps < self.epsilon_warmup_steps:
            # Linear warmup from 0.1 to epsilon_start
            self.epsilon = 0.1 + (self.epsilon_start - 0.1) * (self.steps / self.epsilon_warmup_steps)
        else:
            # Linear decay from epsilon_start to epsilon_end
            decay_progress = min((self.steps - self.epsilon_warmup_steps) / self.epsilon_decay_steps, 1.0)
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): State of shape (C, H, W).
        Returns:
            int: Action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, C, H, W)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()
        return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.ndarray): State of shape (C, H, W).
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state of shape (C, H, W).
            done (bool): Whether the episode is done.
        """
        max_priority = self.memory.total_priority ** self.alpha if self.memory.total_priority > 0 else 1.0
        self.memory.add(max_priority, (state, action, reward, next_state, done))

    def sample(self):
        """
        Sample a batch of transitions using Prioritized Experience Replay.

        Returns:
            Tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
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

        sampling_probabilities = priorities / self.memory.total_priority
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
        """
        Perform a learning step by sampling a batch and updating the model.

        Returns:
            float: Loss value, or None if buffer is not full.
        """
        if self.memory.data_pointer < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.sample()
        self.steps += 1
        self.update_epsilon()  # Update epsilon per step

        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        loss = (weights * F.mse_loss(q_values, target_q_values, reduction='none')).mean()

        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        for idx, error in zip(indices, td_errors):
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
        """
        Save the model and replay buffer.

        Args:
            model_path (str): Path to save the model.
            memory_path (str): Path to save the replay buffer.
        """
        torch.save(self.model.state_dict(), model_path)
        with open(memory_path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, model_path, memory_path):
        """
        Load the model and replay buffer.

        Args:
            model_path (str): Path to load the model.
            memory_path (str): Path to load the replay buffer.
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        with open(memory_path, 'rb') as f:
            self.memory = pickle.load(f)