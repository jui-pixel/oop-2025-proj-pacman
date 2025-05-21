# ai/train.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from game.environment import PacManEnv
from ai.agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json

def train(resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer.pkl", episodes=100):
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = (env.maze.h, env.maze.w, 5)
    action_dim = len(env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device, buffer_size=10000, batch_size=64, lr=5e-4)

    # Load prior experience if resuming
    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"Loaded model from {model_path} and memory from {memory_path}")
    else:
        print("Starting fresh training")

    max_steps = 500  # Reduced for speed
    writer = SummaryWriter()
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            state = next_state
            total_reward += reward
            step += 1
            if loss > 0:
                writer.add_scalar('Loss', loss, episode * max_steps + step)

        episode_rewards.append(total_reward)
        writer.add_scalar('Reward', total_reward, episode)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % 100 == 0:
            agent.save(f"pacman_dqn_{episode+1}.pth", f"replay_buffer_{episode+1}.pkl")

    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)
    writer.close()
    return episode_rewards

if __name__ == "__main__":
    train(resume=True)  # Set resume=True to load prior experience