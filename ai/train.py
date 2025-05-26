"""
訓練 Pac-Man 的 Dueling DQN 代理，負責初始化環境、執行訓練迴圈並記錄指標。
支援從先前模型繼續訓練，並將結果保存到 TensorBoard 和 JSON 檔案。
新增可視化選項，使用 Pygame 即時顯示訓練過程。
"""

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
import pygame
import argparse

def train(resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer.pkl", episodes=1000, visualize=False, render_frequency=10):
    """
    訓練 Dueling DQN 代理，執行指定數量的訓練回合並保存模型。
    僅在 Pac-Man 完成移動後進行訓練。

    Args:
        resume (bool): 是否從先前模型繼續訓練。
        model_path (str): 模型檔案路徑。
        memory_path (str): 記憶緩衝區檔案路徑。
        episodes (int): 訓練回合數。
        visualize (bool): 是否啟用 Pygame 可視化。
        render_frequency (int): 渲染頻率（每多少步渲染一次）。
    """
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    env.render_enabled = visualize
    state_dim = (env.maze.h, env.maze.w, 6)
    action_dim = len(env.action_space)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        buffer_size=100000,
        batch_size=128,
        lr=1e-4,
        epsilon=0.5 if resume else 0.9,  # 如果是從頭開始訓練，使用較高的初始 epsilon
    )

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"Loaded model from {model_path} and memory from {memory_path}")
    else:
        print("Starting fresh training")

    max_steps = 10000
    writer = SummaryWriter()
    episode_rewards = []

    for episode in range(episodes):
        # seed = np.random.randint(0, 10000)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
        env.render_enabled = visualize

        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        last_action = None

        while not done and step < max_steps:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 僅在移動完成時記錄經驗和訓練
            if env.current_action is None:  # 表示移動已完成
                if last_action is not None:  # 確保有上一次動作
                    agent.remember(state, last_action, reward, next_state, done)
                    loss = agent.train()
                    if loss > 0:
                        writer.add_scalar('Loss', loss, episode * max_steps + step)
            
            state = next_state
            total_reward += reward
            last_action = action  # 更新最後動作
            step += 1

            if visualize and step % render_frequency == 0:
                env.render()
        agent.update_epsilon()  # 隨著訓練進行，逐漸減少探索率
        episode_rewards.append(total_reward)
        writer.add_scalar('Reward', total_reward, episode)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()
    if visualize:
        pygame.quit()
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 Pac-Man Dueling DQN 代理")
    parser.add_argument('-v', '--visualize', type=str, default='False', help="是否啟用 Pygame 可視化 ('True' 或 'False')")
    args = parser.parse_args()

    visualize = args.visualize.lower() == 'true'
    for _ in range(10):
        train(resume=True, visualize=visualize, render_frequency=10)