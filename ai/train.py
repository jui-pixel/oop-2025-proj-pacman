# ai/train.py
"""
訓練 Pac-Man 的 DQN 代理，負責初始化環境、執行訓練迴圈並記錄指標。
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
import pygame  # 引入 Pygame 用於可視化

def train(resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer.pkl", episodes=1000, visualize=False, render_frequency=10):
    """
    訓練 DQN 代理，執行指定數量的訓練回合並保存模型與記憶緩衝區。

    Args:
        resume (bool): 是否從先前模型繼續訓練，預設為 False。
        model_path (str): 模型檔案路徑，預設為 "pacman_dqn_final.pth"。
        memory_path (str): 記憶緩衝區檔案路徑，預設為 "replay_buffer.pkl"。
        episodes (int): 訓練回合數，預設為 1000。
        visualize (bool): 是否啟用 Pygame 可視化，預設為 False。
        render_frequency (int): 渲染頻率（每多少步渲染一次），預設為 10。

    Returns:
        List[float]: 每個回合的總獎勵列表。
    """
    # 初始化 Pac-Man 遊戲環境
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    env.render_enabled = visualize  # 設置可視化開關
    state_dim = (env.maze.h, env.maze.w, 6)
    action_dim = len(env.action_space)

    # 選擇計算設備（優先使用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 初始化 DQN 代理
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        buffer_size=100000,
        batch_size=128,
        lr=1e-5 if resume else 1e-4
    )

    # 載入先前模型
    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        agent.epsilon = 0.3
        print(f"Loaded model from {model_path} and memory from {memory_path}")
    else:
        print("Starting fresh training")

    max_steps = 5000
    writer = SummaryWriter()
    episode_rewards = []

    # 主訓練迴圈
    for episode in range(episodes):
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=seed)
        env.render_enabled = visualize  # 為新環境設置可視化開關

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

            # 記錄損失
            if loss > 0:
                writer.add_scalar('Loss', loss, episode * max_steps + step)

            # 可視化（根據頻率渲染）
            if visualize and step % render_frequency == 0:
                env.render()

        episode_rewards.append(total_reward)
        writer.add_scalar('Reward', total_reward, episode)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Seed = {seed}")

    # 保存模型和記憶緩衝區
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")

    # 保存回合獎勵
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()
    # 如果啟用了可視化，關閉 Pygame
    if visualize:
        pygame.quit()
    return episode_rewards

if __name__ == "__main__":
    for _ in range(10):  # 執行10次訓練
        train(resume=True, visualize=True, render_frequency=10)  # 啟用可視化，每 10 步渲染一次