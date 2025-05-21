# ai/train.py
import torch
from game.environment import PacManEnv
from ai.agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def train():
    # Initialize environment with config values
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = (env.maze.h, env.maze.w, 5)  # (height, width, channels)
    action_dim = len(env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device, buffer_size=10000, batch_size=32, lr=1e-4)

    episodes = 1000
    max_steps = 1000
    writer = SummaryWriter()  # For TensorBoard logging
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
            agent.save(f"pacman_dqn_{episode+1}.pth")

    agent.save("pacman_dqn_final.pth")
    writer.close()
    return episode_rewards

if __name__ == "__main__":
    train()