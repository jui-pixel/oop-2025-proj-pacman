# parallel_train.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from game.environment import PacManEnv
from ai.agent import DQNAgent, PrioritizedReplayBuffer  # 假設使用自定義緩衝區
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import argparse
from multiprocessing import Process, Queue, Manager, Lock
import time
import gc

def train_worker(shared_buffer, lock, episode_queue, reward_queue, resume=False, model_path="pacman_dqn_final.pth", memory_path="replay_buffer_final.pkl", episodes=1000, worker_id=0):
    """
    並行訓練 Dueling DQN 代理的單一工作進程，共享記憶緩衝區。

    Args:
        shared_buffer (list): 共享的經驗回放緩衝區。
        lock (Lock): 用於同步訪問共享緩衝區的鎖。
        episode_queue (Queue): 用於傳遞當前訓練集數。
        reward_queue (Queue): 用於傳回每局獎勵。
        resume (bool): 是否從先前模型繼續訓練。
        model_path (str): 模型檔案路徑。
        memory_path (str): 記憶緩衝區檔案路徑。
        episodes (int): 訓練回合數。
        worker_id (int): 工作進程的 ID。
    """
    log_dir = f"runs/worker_{worker_id}_{int(time.time())}"
    writer = SummaryWriter(log_dir)

    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED + worker_id)
    state_dim = (env.maze.h, env.maze.w, 6)
    action_dim = len(env.action_space)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Worker {worker_id} 使用裝置：{device}")

    # 每個 worker 獨立創建 agent，但共享緩衝區
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        buffer_size=10000,  # 本地緩衝區用於臨時儲存
        batch_size=32,
        lr=5e-3,
        epsilon=1.0,
    )

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"Worker {worker_id} Loaded model from {model_path} and memory from {memory_path}")
    else:
        print(f"Worker {worker_id} Starting fresh training")

    max_steps = 10000
    episode_rewards = []

    for episode in range(episodes):
        episode_queue.put(episode)
        env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED + worker_id + episode)
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        last_action = None

        while not done and step < max_steps:
            if env.pacman.move_towards_target:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                if env.current_action is None:
                    if last_action is not None:
                        # 臨時儲存經驗
                        experience = (state, last_action, reward, next_state, done)
                        with lock:
                            shared_buffer.append(experience)  # 同步添加到共享緩衝區

                state = next_state
                total_reward += reward
                last_action = action
                step += 1

        episode_rewards.append(total_reward)
        writer.add_scalar('Reward', total_reward, episode)
        print(f"Worker {worker_id}, Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        agent.update_epsilon()

        del state, next_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agent.save(model_path_worker, memory_path)
    reward_queue.put((worker_id, episode_rewards))
    writer.close()

def train_parallel(num_workers=4, resume=False, episodes=1000, buffer_size=100000):
    """
    並行執行多個訓練進程，共享記憶緩衝區。

    Args:
        num_workers (int): 並行工作進程數。
        resume (bool): 是否從先前模型繼續訓練。
        episodes (int): 每個進程的訓練回合數。
        buffer_size (int): 共享緩衝區的大小。
    """
    manager = Manager()
    shared_buffer = manager.list()  # 共享的經驗回放緩衝區
    lock = Lock()  # 同步鎖
    episode_queue = manager.Queue()
    reward_queue = manager.Queue()

    processes = []
    for i in range(num_workers):
        p = Process(target=train_worker, args=(shared_buffer, lock, episode_queue, reward_queue, resume, f"pacman_dqn_final.pth", f"replay_buffer_final.pkl", episodes, i))
        processes.append(p)
        p.start()

    # 主進程負責訓練（從共享緩衝區抽樣）
    master_agent = None
    all_rewards = []
    for _ in range(num_workers):
        worker_id, rewards = reward_queue.get()
        all_rewards.extend(rewards)

        # 初始化或更新主代理
        if master_agent is None:
            env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
            state_dim = (env.maze.h, env.maze.w, 6)
            action_dim = len(env.action_space)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            master_agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                buffer_size=buffer_size,
                batch_size=32,
                lr=5e-3,
                epsilon=1.0,
            )
            if resume and os.path.exists("pacman_dqn_final.pth"):
                master_agent.load("pacman_dqn_final.pth", "replay_buffer_final.pkl")

        # 從共享緩衝區抽樣並訓練主代理
        if len(shared_buffer) >= master_agent.batch_size:
            with lock:
                batch = shared_buffer[:master_agent.batch_size]  # 抽樣一批經驗
                shared_buffer[:] = shared_buffer[master_agent.batch_size:]  # 移除已抽樣數據
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.stack([torch.FloatTensor(s) for s in states]).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.stack([torch.FloatTensor(ns) for ns in next_states]).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)
            loss = master_agent.train_step(states, actions, rewards, next_states, dones)
            if loss is not None and loss > 0:
                print(f"Master Agent Training Loss: {loss.item():.4f}")

        # 定期同步模型參數到所有 worker
        if len(shared_buffer) % 100 == 0 and len(shared_buffer) > 0:
            master_agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
            for p in processes:
                if p.is_alive():
                    # 假設 agent 支持動態加載（需在 DQNAgent 中實現）
                    p.terminate()  # 終止舊進程
                    p = Process(target=train_worker, args=(shared_buffer, lock, episode_queue, reward_queue, True, "pacman_dqn_final.pth", "replay_buffer_final.pkl", episodes, worker_id))
                    p.start()
                    processes[worker_id] = p

    for p in processes:
        p.join()

    if os.path.exists("episode_rewards.json"):
        with open("episode_rewards.json", "r") as f:
            try:
                old_rewards = json.load(f)
                if isinstance(old_rewards, list):
                    all_rewards = old_rewards + all_rewards
            except json.JSONDecodeError:
                print("Warning: episode_rewards.json is corrupted, starting fresh.")
    with open("episode_rewards.json", "w") as f:
        json.dump(all_rewards, f)

    gc.collect()
    return all_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="平行訓練 Pac-Man Dueling DQN 代理")
    parser.add_argument('-n', '--num_workers', type=int, default=4, help="並行工作進程數")
    args = parser.parse_args()

    num_workers = args.num_workers
    for _ in range(100 // num_workers):
        train_parallel(num_workers=num_workers, resume=True, episodes=100, buffer_size=100000)