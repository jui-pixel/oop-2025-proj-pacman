# ai/train.py
import torch
import os
import argparse
import numpy as np
import json
import pygame
import logging
from environment import PacManEnv
from agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer_final.pkl", episodes=1000, visualize=False, render_frequency=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Training started with device: {device}")

    try:
        env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED, visualize=visualize)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {str(e)}")
        raise RuntimeError(f"Failed to initialize environment: {str(e)}")

    state_dim = env.state_shape
    action_dim = 4

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device, gamma=0.99, n_step=1)

    if resume and os.path.exists(model_path) and os.path.exists(memory_path):
        agent.load(model_path, memory_path)
        print(f"Loaded model from {model_path}")
        logger.info(f"Loaded model from {model_path}")
    elif resume:
        print(f"Warning: Model or memory file not found. Starting fresh.")
        logger.warning("Model or memory file not found. Starting fresh.")

    writer = SummaryWriter()
    episode_rewards = []

    for episode in range(episodes):
        try:
            state = env.reset()
        except Exception as e:
            logger.error(f"Reset failed in episode {episode+1}: {str(e)}")
            raise RuntimeError(f"Reset failed in episode {episode+1}: {str(e)}")

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.choose_action(state)
            try:
                next_state, reward, done, info = env.step(action)
            except Exception as e:
                logger.error(f"Step failed in episode {episode+1}, step {step_count+1}: {str(e)}")
                raise RuntimeError(f"Step failed in episode {episode+1}, step {step_count+1}: {str(e)}")

            agent.store_transition(state, action, reward, next_state, done)
            state = next_state

            loss = agent.learn()
            if loss is not None:
                writer.add_scalar('Loss/Agent', loss, agent.steps)

            total_reward += reward
            step_count += 1

            if visualize and step_count % render_frequency == 0:
                env.render()

            if done:
                logger.info(f"Episode {episode+1} ended: Reward={total_reward:.2f}, Lives={info['lives']}, GameOver={info['game_over']}")

        episode_rewards.append(total_reward)
        writer.add_scalar('Reward/Episode', total_reward, episode)
        writer.add_scalar('Epsilon/Episode', agent.epsilon, episode)
        writer.add_scalar('Beta/Episode', agent.beta, episode)
        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Beta: {agent.beta:.3f}, Steps: {step_count}")
        logger.info(f"Episode {episode+1} completed: Reward={total_reward:.2f}, Steps={step_count}")

        if (episode + 1) % 10 == 0:
            agent.save(model_path, memory_path)
            logger.info(f"Saved model at episode {episode+1}")

    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()
    if visualize:
        env.close()
    logger.info("Training completed")
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man Dueling DQN Agent")
    parser.add_argument('--visualize', type=bool, default=False, help='Enable Pygame visualization')
    parser.add_argument('--resume', action='store_true', help='Resume from previous model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--render_frequency', type=int, default=10, help='Render frequency')
    args = parser.parse_args()

    train(resume=args.resume, episodes=args.episodes, visualize=args.visualize, render_frequency=args.render_frequency)