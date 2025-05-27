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
import multiprocessing as mp
from queue import Empty

def worker_process(env_id, state_queue, action_queue, reward_queue, done_queue, width, height, seed):
    """子進程運行單個環境，執行動作並返回經驗"""
    try:
        env = PacManEnv(width=width, height=height, seed=seed + env_id)  # 每個環境不同種子
        env.render_enabled = False  # 禁用可視化
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        max_steps = 10000
        last_action = None

        while not done and step < max_steps:
            state_queue.put((env_id, state))  # 將狀態發送到主進程
            try:
                if env.pacman.move_towards_target(env.maze):
                    action = action_queue.get(timeout=1)  # 從主進程獲取動作
                    next_state, reward, done, _ = env.step(action)
                    # print(f"Env {env_id}: Step {step}, action {action}, reward {reward}, done {done}")
                    if env.current_action is None:  # 僅在移動完成時記錄
                        reward_queue.put((env_id, state, last_action, reward, next_state, done))
                    state = next_state
                    total_reward += reward
                    last_action = action
                    step += 1
            except Empty:
                continue
        done_queue.put((env_id, total_reward))
        # print(f"Env {env_id}: Sent done, total reward {total_reward}")
    finally:
        pygame.quit()
    
    
    
def train_parallel(resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer_final.pkl", episodes=2000, num_envs=4):
    """
    並行訓練 Dueling DQN 代理，使用多個環境同時收集經驗。

    Args:
        resume (bool): 是否從先前模型繼續訓練。
        model_path (str): 模型檔案路徑。
        memory_path (str): 記憶緩衝區檔案路徑。
        episodes (int): 總訓練回合數。
        num_envs (int): 並行環境數量。
    """
    state_dim = (MAZE_HEIGHT, MAZE_WIDTH, 6)
    action_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        buffer_size=50000, 
        batch_size=128,
        lr=5e-4, 
        epsilon=0.5 
    )

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        agent.epsilon = 0.01
        print(f"載入模型：{model_path}，記憶緩衝區：{memory_path}")
    else:
        print("開始全新訓練")

    writer = SummaryWriter()
    episode_rewards = []

    # 初始化多進程隊列
    state_queue = mp.Queue(maxsize=1000)
    action_queue = mp.Queue(maxsize=1000)
    reward_queue = mp.Queue(maxsize=1000)
    done_queue = mp.Queue(maxsize=1000)

    for episode in range(0, episodes, num_envs):
        # 啟動多個環境進程
        processes = []
        for env_id in range(num_envs):
            if episode + env_id < episodes:
                p = mp.Process(target=worker_process, args=(env_id, state_queue, action_queue, reward_queue, done_queue, MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED))
                p.start()
                processes.append(p)

        # 主進程處理動作選擇和訓練
        active_envs = len(processes)
        env_rewards = {}
        while active_envs > 0:
            # 先檢查 done_queue
            while True:
                try:
                    env_id, total_reward = done_queue.get_nowait()
                    env_rewards[env_id] = total_reward
                    active_envs -= 1
                except Empty:
                    break
            # 再檢查 state_queue
            try:
                env_id, state = state_queue.get(timeout=1)
                action = agent.get_action(state)
                action_queue.put(action)
                ...
            except Empty:
                continue

        # 記錄每個環境的獎勵
        for env_id in range(num_envs):
            if env_id in env_rewards:
                episode_rewards.append(env_rewards[env_id])
                writer.add_scalar('Reward', env_rewards[env_id], episode + env_id)
                print(f"Episode {episode + env_id + 1}/{episodes}, Total Reward: {env_rewards[env_id]:.2f}, Epsilon: {agent.epsilon:.3f}")
        agent.update_epsilon()

        # 每 100 局保存模型，記憶緩衝區每 500 局保存
        if episode % 500 == 0 and episode > 0:
            agent.save(f"pacman_dqn_ep{episode}.pth", f"replay_buffer_ep{episode}.pkl")
        elif episode % 100 == 0:
            agent.save(f"pacman_dqn_ep{episode}.pth", None)  # 僅保存模型
        
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()
        for q in [state_queue, action_queue, reward_queue, done_queue]:
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break
            q.close()  # 關閉隊列，釋放資源
            q.join_thread() # 等待隊列清空
        

    # 最終保存
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()
    pygame.quit()
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="並行訓練 Pac-Man Dueling DQN 代理")
    parser.add_argument('-n', '--num_envs', type=int, default=4, help="並行環境數量")
    args = parser.parse_args()

    train_parallel(resume=True, num_envs=args.num_envs)