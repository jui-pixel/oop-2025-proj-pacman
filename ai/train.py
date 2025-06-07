import os
import argparse
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from gym.vector import SyncVectorEnv
from environment import PacManEnv
from agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

def make_env(seed=MAZE_SEED):
    """Create a single PacManEnv environment."""
    def _env():
        return PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=seed)
    return _env

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl", 
          episodes=1000, num_envs=4, early_stop_reward=2000):
    """
    Train a basic DQN agent with parallel environments and early stopping, with reduced terminal output frequency.

    Args:
        resume (bool): Whether to resume from a previous model.
        model_path (str): Path to save/load the model.
        memory_path (str): Path to save/load the replay buffer.
        episodes (int): Number of training episodes.
        num_envs (int): Number of parallel environments.
        early_stop_reward (float): Average reward threshold for early stopping.

    Returns:
        list: Total rewards for each episode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Initialize parallel environments
    envs = SyncVectorEnv([make_env(seed=MAZE_SEED + i) for i in range(num_envs)])
    state_dim = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n

    # Initialize agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"Loaded model from {model_path}")

    writer = SummaryWriter()
    episode_rewards = []
    recent_rewards = []

    # Training loop
    states, _ = envs.reset()
    for episode in range(episodes):
        total_rewards = np.zeros(num_envs)
        step_counts = np.zeros(num_envs, dtype=int)
        dones = np.zeros(num_envs, dtype=bool)

        while not dones.all():
            actions = [agent.choose_action(state) for state in states]
            next_states, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)

            for i in range(num_envs):
                agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
                total_rewards[i] += rewards[i]
                step_counts[i] += 1

            loss = agent.learn()
            if loss is not None:
                writer.add_scalar('Loss', loss, agent.steps)

            states = next_states

        # Record results
        avg_reward = np.mean(total_rewards)
        episode_rewards.append(avg_reward)
        recent_rewards.append(avg_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                  f"Steps: {int(np.mean(step_counts))}, Epsilon: {agent.epsilon:.3f}")

        # Log to TensorBoard
        writer.add_scalar('Reward/Average', avg_reward, episode)
        for i in range(num_envs):
            writer.add_scalar(f'Reward/Env{i+1}', total_rewards[i], episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)

        # Save model periodically
        if (episode + 1) % 10 == 0:
            agent.save(model_path, memory_path)
            print(f"Saved model at episode {episode+1}")

        # Early stopping check
        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= early_stop_reward:
            print(f"Early stopping: Avg reward {np.mean(recent_rewards[-100:]):.2f} >= {early_stop_reward}")
            break

    # Save final model and rewards
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()
    envs.close()
    print("Training completed")
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man DQN Agent")
    parser.add_argument('--resume', action='store_true', help='Resume from previous model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--early_stop_reward', type=float, default=2000, help='Reward threshold for early stopping')
    args = parser.parse_args()

    train(resume=args.resume, episodes=args.episodes, num_envs=args.num_envs, 
          early_stop_reward=args.early_stop_reward)