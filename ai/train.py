import torch
import os
import argparse
import numpy as np
import json
import logging
from environment import PacManEnv
from agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
from torch.utils.tensorboard import SummaryWriter
from gym.vector import SyncVectorEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env(seed=MAZE_SEED):
    def _env():
        env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=seed)
        return env
    return _env

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl", episodes=1000, num_envs=4, early_stop_reward=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    try:
        envs = SyncVectorEnv([make_env(seed=MAZE_SEED+i) for i in range(num_envs)])
    except Exception as e:
        logger.error(f"Failed to initialize environments: {str(e)}")
        raise RuntimeError(f"Failed to initialize environments: {str(e)}")

    state_dim = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        logger.info(f"Loaded model from {model_path}")

    writer = SummaryWriter()
    episode_rewards = []
    recent_rewards = []

    states, _ = envs.reset()
    for episode in range(episodes):
        total_rewards = np.zeros(num_envs)
        step_counts = np.zeros(num_envs, dtype=int)
        dones = np.zeros(num_envs, dtype=bool)

        while not dones.all():
            actions = [agent.choose_action(state) for state in states]
            try:
                next_states, rewards, terminated, truncated, infos = envs.step(actions)
                dones = np.logical_or(terminated, truncated)
            except Exception as e:
                logger.error(f"Step failed: {str(e)}")
                raise RuntimeError(f"Step failed: {str(e)}")

            for i in range(num_envs):
                agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
                total_rewards[i] += rewards[i]
                step_counts[i] += 1

            loss = agent.learn()
            if loss is not None:
                writer.add_scalar('Loss/Agent', loss, agent.steps)

            states = next_states

        for i in range(num_envs):
            logger.info(f"Episode {episode+1}, Env {i+1}: Reward={total_rewards[i]:.2f}, Steps={step_counts[i]}")
            episode_rewards.append(total_rewards[i])
            recent_rewards.append(total_rewards[i])
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            writer.add_scalar(f'Reward/Env{i+1}', total_rewards[i], episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Beta', agent.beta, episode)

        avg_reward = np.mean(recent_rewards[-100:]) if recent_rewards else 0
        print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        if episode % 10 == 0:
            agent.save(model_path, memory_path)
            logger.info(f"Saved model at episode {episode+1}")

        if len(recent_rewards) >= 100 and avg_reward >= early_stop_reward:
            logger.info(f"Early stopping: Avg reward {avg_reward:.2f} >= {early_stop_reward}")
            break

    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()
    envs.close()
    logger.info("Training completed")
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man Dueling DQN Agent")
    parser.add_argument('--resume', action='store_true', help='Resume from previous model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--early_stop_reward', type=float, default=500, help='Reward threshold for early stopping')
    args = parser.parse_args()

    train(resume=args.resume, episodes=args.episodes, num_envs=args.num_envs, early_stop_reward=args.early_stop_reward)