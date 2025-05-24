# ai/train.py
"""
訓練 Pac-Man 的 DQN 代理，負責初始化環境、執行訓練迴圈並記錄指標。
支援從先前模型繼續訓練，並將結果保存到 TensorBoard 和 JSON 檔案。
"""
import sys
import os
# 將專案根目錄添加到 Python 路徑，確保模組可正確導入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from game.environment import PacManEnv
from ai.agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json

def train(resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer.pkl", episodes=1000):
    """
    訓練 DQN 代理，執行指定數量的訓練回合並保存模型與記憶緩衝區。

    Args:
        resume (bool): 是否從先前模型繼續訓練，預設為 False。
        model_path (str): 模型檔案路徑，預設為 "pacman_dqn_final.pth"。
        memory_path (str): 記憶緩衝區檔案路徑，預設為 "replay_buffer.pkl"。
        episodes (int): 訓練回合數，預設為 1000。

    Returns:
        List[float]: 每個回合的總獎勵列表。
    """
    # 初始化 Pac-Man 遊戲環境
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = (env.maze.h, env.maze.w, 6)  # 狀態維度：(高度, 寬度, 6 通道)
    action_dim = len(env.action_space)  # 動作維度：4（上、下、左、右）
    
    # 選擇計算設備（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置：", device)
    
    # 初始化 DQN 代理
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        buffer_size=50000,  # 記憶緩衝區大小
        batch_size=128,      # 批次大小
        lr=1e-5 if resume else 1e-4 # 學習率
    )
    
    # 如果 resume=True 且模型檔案存在，載入先前訓練的模型和記憶緩衝區
    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)  # 載入記憶緩衝區
        agent.epsilon = 0.3  # 提高 epsilon 以增加探索
        print(f"Loaded model from {model_path} and memory from {memory_path}")
    else:
        print("Starting fresh training")

    
    max_steps = 1000  # 每個回合的最大步數
    writer = SummaryWriter()  # 初始化 TensorBoard 記錄器
    episode_rewards = []  # 記錄每個回合的總獎勵
    
    # 訓練迴圈
    for episode in range(episodes):
        if(1):  # 每個回合都重新生成環境
            seed = np.random.randint(0, 10000)  # 隨機種子
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=seed)  # 每次創建新環境
        state = env.reset()  # 重置環境
        total_reward = 0
        done = False
        step = 0
        
        # 單個回合的步進迴圈
        while not done and step < max_steps:
            action = agent.get_action(state)  # 根據當前狀態選擇動作
            next_state, reward, done, _ = env.step(action)  # 執行動作
            agent.remember(state, action, reward, next_state, done)  # 儲存經驗
            loss = agent.train()  # 訓練模型
            state = next_state
            total_reward += reward
            step += 1
            
            # 記錄損失（如果有）
            if loss > 0:
                writer.add_scalar('Loss', loss, episode * max_steps + step)
        
        episode_rewards.append(total_reward)
        writer.add_scalar('Reward', total_reward, episode)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # # 每 100 回合保存一次模型和記憶緩衝區
        # if (episode + 1) % 100 == 0:
        #     agent.save(f"pacman_dqn_{episode+1}.pth", f"replay_buffer_{episode+1}.pkl")
    
    # 保存最終模型和記憶緩衝區
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    
    # 保存回合獎勵到 JSON 檔案
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)
    
    writer.close()  # 關閉 TensorBoard 記錄器
    return episode_rewards

if __name__ == "__main__":
    train(resume=True)  # 執行訓練，預設從先前模型繼續