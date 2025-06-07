import os
import argparse
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from environment import PacManEnv
from agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl", 
          episodes=1000, early_stop_reward=2000):
    """
    Train a basic DQN agent with a single environment and early stopping.

    Args:
        resume (bool): Whether to resume from a previous model.
        model_path (str): Path to save/load the model.
        memory_path (str): Path to save/load the replay buffer.
        episodes (int): Number of training episodes.
        early_stop_reward (float): Average reward threshold for early stopping.

    Returns:
        list: Total rewards for each episode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Initialize single environment
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    pacman = env.pacman
    # Initialize agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"Loaded model from {model_path}")

    writer = SummaryWriter()
    episode_rewards = []
    recent_rewards = []

    # Training loop
    state, _ = env.reset()
    for episode in range(episodes):
        total_reward = 0
        step_count = 0
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # print(info)  # Debugging info
            if info['valid_step'] == True:  # 僅在移動完成時存儲和學習
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    writer.add_scalar('Loss', loss, agent.steps)
                # print(f"reward = {reward}, action = {action}, loss = {loss}")
                total_reward += reward
                step_count += 1
            state = next_state

        # Record results
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Print progress every episode
        if (episode + 1) % 1 == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, "
                  f"Steps: {step_count}, Epsilon: {agent.epsilon:.3f}")

        # Log to TensorBoard
        writer.add_scalar('Reward', total_reward, episode)
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
    env.close()
    print("Training completed")
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man DQN Agent")
    parser.add_argument('--resume', action='store_true', help='Resume from previous model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--early_stop_reward', type=float, default=2000, help='Reward threshold for early stopping')
    args = parser.parse_args()
    train(resume=args.resume, episodes=args.episodes, early_stop_reward=args.early_stop_reward)