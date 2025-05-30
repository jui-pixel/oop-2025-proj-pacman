# ai/train.py
"""
訓練 Pac-Man 的 Dueling DQN 代理，負責初始化環境、執行訓練迴圈並記錄指標。
支援從先前模型繼續訓練，並將結果保存到 TensorBoard 和 JSON 檔案。
新增可視化選項，使用 Pygame 即時顯示訓練過程。
"""

import sys
import os
# 修正 sys.path.insert 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ai.environment import PacManEnv # 確保正確導入
from ai.agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import pygame
import argparse

def train(resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer_final.pkl", episodes=1000, visualize=False, render_frequency=10):
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
    # 檢查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    
    # 根據 env.state_shape 定義 state_dim
    # state_shape 是 (channels, height, width)，agent 需要 (height, width, channels)
    # 所以這裡需要轉換回來，或者讓 agent 直接接受 (C, H, W)
    # 由於你的 dqn.py 中的 DuelingDQN 接收 (H, W, C) 但實際處理為 (C, H, W)
    # 這裡agent的state_dim也應該匹配DuelingDQN的input_dim
    state_dim = env.state_shape # 現在 env._get_state() 已經轉置為 (C, H, W)
    action_dim = 4 # 上下左右

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device,
                     gamma=0.99, n_step=1) # 傳遞 gamma 和 n_step

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"從 {model_path} 載入模型和記憶緩衝區。")
    else:
        print("開始新的訓練。")

    writer = SummaryWriter() # TensorBoard 記錄器
    episode_rewards = [] # 儲存每個回合的總獎勵

    for episode in range(episodes):
        state = env.reset() # 重置環境，獲取初始狀態
        done = False
        total_reward = 0
        step_count = 0 # 記錄當前回合的步數

        while not done:
            action = agent.choose_action(state) # 代理選擇動作
            next_state, reward, done, info = env.step(action) # 執行動作，獲取下一個狀態、獎勵和結束標誌

            agent.store_transition(state, action, reward, next_state, done) # 儲存經驗

            state = next_state # 更新當前狀態

            # 只有當 Pac-Man 移動完成後才進行學習
            # 這裡簡單判斷 Pac-Man 是否仍在移動（例如，如果它的 current_x, current_y 與 grid_x, grid_y 不完全對齊）
            # 或者更簡單，每 N 步學習一次
            # 你的agent.learn() 已經包含了 batch_size 的檢查，所以可以每一步都嘗試學習
            loss = agent.learn() # 訓練模型
            if loss is not None:
                writer.add_scalar('Loss/Agent', loss, agent.steps) # 記錄損失

            total_reward += reward
            step_count += 1

            if visualize and step_count % render_frequency == 0:
                env.render()

        episode_rewards.append(total_reward)
        writer.add_scalar('Reward/Episode', total_reward, episode) # 修改為 'Reward/Episode'
        writer.add_scalar('Epsilon/Episode', agent.epsilon, episode) # 記錄 epsilon
        writer.add_scalar('Beta/Episode', agent.beta, episode) # 記錄 beta
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Beta: {agent.beta:.3f}, Steps: {step_count}")
        agent.update_epsilon()  # 隨著訓練進行，逐漸減少探索率
        
        # 每 10 個回合保存一次模型
        if (episode + 1) % 10 == 0:
            agent.save(model_path, memory_path)      

    # 訓練結束後保存最終模型
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    
    # 處理歷史獎勵數據
    if os.path.exists("episode_rewards.json"):
        with open("episode_rewards.json", "r") as f:
            try:
                old_rewards = json.load(f)
                if isinstance(old_rewards, list):
                    episode_rewards.extend(old_rewards) # 擴展舊的獎勵
            except json.JSONDecodeError:
                print("Warning: episode_rewards.json is corrupted, starting fresh.")
                
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f) # 保存所有獎勵
    
    writer.close()
    if visualize:
        env.close() # 關閉 Pygame 視窗
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 Pac-Man Dueling DQN 代理")
    parser.add_argument('-v', '--visualize', type=bool, default=False, help='是否啟用 Pygame 可視化')
    parser.add_argument('-r', '--resume', type=bool, default=False, help='是否從先前模型繼續訓練')
    parser.add_argument('-e', '--episodes', type=int, default=1000, help='訓練回合數')
    parser.add_argument('-f', '--render_frequency', type=int, default=10, help='渲染頻率（每多少步渲染一次）')

    args = parser.parse_args()

    train(resume=args.resume, episodes=args.episodes, visualize=args.visualize, render_frequency=args.render_frequency)