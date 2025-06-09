# ai/train.py
import os
import argparse
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from environment import PacManEnv
from agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
import random

def collect_expert_data(env, agent, num_episodes=100):
    """
    使用規則基礎 AI 收集專家數據，用於模仿學習預訓練。

    原理：
    - 專家數據由規則基礎的 AI（例如基於路徑規劃的 Pac-Man 策略）生成，提供高質量的 (狀態, 動作) 對。
    - 這些數據用於初始化 DQN 模型，使其學習專家行為，減少早期隨機探索的低效性。
    - 每個回合中，環境執行專家動作，記錄有效步驟的狀態和動作，並存入代理的回放緩衝區。
    - 有效步驟由環境的 info['valid_step'] 標誌確定，確保只記錄有意義的轉換。

    Args:
        env (PacManEnv): Pac-Man 環境，提供狀態、動作和獎勵。
        agent (DQNAgent): DQN 代理，用於儲存轉換數據。
        num_episodes (int): 收集專家數據的回合數。

    Returns:
        list: 包含 (state, action) 的專家數據列表，用於預訓練。
    """
    expert_data = []  # 儲存專家數據
    for episode in range(num_episodes):
        # 重置環境，使用 episode 作為隨機種子以確保多樣性
        state, _ = env.reset(random_spawn_seed=episode)
        done = False  # 回合是否結束
        while not done:
            # 獲取專家動作（由環境的規則基礎 AI 提供）
            action = env.get_expert_action()
            # 執行動作，獲取下一狀態、獎勵和結束標誌
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # 回合結束條件
            if info.get('valid_step', False):  # 僅記錄有效步驟
                expert_data.append((state, action))  # 儲存 (狀態, 動作) 對
                # 將轉換存入代理的回放緩衝區
                agent.store_transition(state, action, reward, next_state, done)
            state = next_state  # 更新當前狀態
        print(f"專家回合 {episode + 1}/{num_episodes} 完成")
    return expert_data

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl", 
          episodes=1000, early_stop_reward=2000, pretrain_episodes=100):
    """
    訓練 DQN 代理，結合模仿學習預訓練和早期停止機制。

    原理：
    - 訓練過程基於深度 Q 學習（DQN），通過反覆試驗優化代理在 Pac-Man 環境中的策略。
    - 主要步驟包括：
      1. **模仿學習預訓練**：使用專家數據初始化模型，通過交叉熵損失學習專家行為。
      2. **探索與利用**：結合專家策略（概率逐漸衰減）和 DQN 的 Noisy DQN 探索，平衡探索與利用。
      3. **經驗回放**：從回放緩衝區採樣轉換，計算 TD 誤差並更新模型。
      4. **早期停止**：當最近 100 回合的平均獎勵達到閾值時停止訓練，防止過擬合。
      5. **記錄與保存**：使用 TensorBoard 記錄損失和獎勵，定期保存模型和回放緩衝區。
    - 獎勵計算：總獎勵為每回合累積的環境獎勵，反映代理的表現。
    - 專家概率衰減公式：expert_prob = expert_prob_start - (expert_prob_start - expert_prob_end) * (steps / decay_steps)

    Args:
        resume (bool): 是否從之前的模型繼續訓練。
        model_path (str): 模型保存/載入路徑。
        memory_path (str): 回放緩衝區保存/載入路徑。
        episodes (int): 訓練的總回合數。
        early_stop_reward (float): 平均獎勵閾值，若達到則提前停止。
        pretrain_episodes (int): 模仿學習預訓練的回合數。

    Returns:
        list: 每個回合的總獎勵列表。
    """
    # 選擇計算設備（優先使用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練設備：{device}")

    # 初始化 Pac-Man 環境
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = env.observation_space.shape  # 狀態空間維度
    action_dim = env.action_space.n  # 動作空間維度
    pacman = env.pacman  # Pac-Man 對象
    # 初始化 DQN 代理
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    # 如果 resume 為真且模型存在，則載入模型和回放緩衝區
    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        agent.expert_prob = 0.01
        print(f"從 {model_path} 載入模型")

    # 如果不繼續訓練，執行模仿學習預訓練
    if not resume:
        print(f"收集 {pretrain_episodes} 回合的專家數據進行預訓練...")
        expert_data = collect_expert_data(env, agent, pretrain_episodes)
        # 使用專家數據進行預訓練，步數固定為 10000
        agent.pretrain(expert_data, pretrain_steps=10000)

    # 初始化 TensorBoard 用於記錄訓練指標
    writer = SummaryWriter()
    episode_rewards = []  # 記錄每個回合的總獎勵
    recent_rewards = []  # 記錄最近 100 回合的獎勵，用於早期停止

    # 訓練主循環
    state, _ = env.reset()
    for episode in range(episodes):
        total_reward = 0  # 當前回合總獎勵
        step_count = 0  # 當前回合步數
        done = False  # 回合是否結束
        state, _ = env.reset(random_spawn_seed=episode)  # 重置環境，使用 episode 作為種子
        agent.model.reset_noise()  # 重置 Noisy DQN 的噪聲
        while not done:
            # 根據專家概率決定是否使用專家動作
            if random.random() < agent.expert_prob:
                action = env.get_expert_action()  # 專家動作
                expert_action = True
            else:
                action = agent.choose_action(state)  # DQN 動作
                expert_action = False
            # 執行動作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get('valid_step', False):  # 僅處理有效步驟
                # 儲存轉換到回放緩衝區
                agent.store_transition(state, action, reward, next_state, done)
                # 執行學習步驟
                loss = agent.learn(expert_action=expert_action)
                if loss is not None:
                    writer.add_scalar('Loss', loss, agent.steps)  # 記錄損失
                total_reward += reward  # 累積獎勵
                step_count += 1  # 增加步數
            state = next_state  # 更新狀態

        episode_rewards.append(total_reward)  # 記錄回合獎勵
        recent_rewards.append(total_reward)  # 更新最近獎勵列表
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)  # 移除最早的獎勵

        # 每回合打印進度
        if (episode + 1) % 1 == 0:
            print(f"回合 {episode+1}/{episodes}，獎勵：{total_reward:.2f}，"
                  f"步數：{step_count}，專家概率：{agent.expert_prob:.3f}")

        # 記錄 TensorBoard 數據
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Expert Probability', agent.expert_prob, episode)

        # 每 10 回合保存模型
        if (episode + 1) % 10 == 0:
            agent.save(model_path, memory_path)
            print(f"在回合 {episode+1} 保存模型")

        # 早期停止檢查
        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= early_stop_reward:
            print(f"早期停止：最近 100 回合平均獎勵 {np.mean(recent_rewards[-100:]):.2f} >= {early_stop_reward}")
            break

    # 保存最終模型和回放緩衝區
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    # 保存回合獎勵數據
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()  # 關閉 TensorBoard
    env.close()  # 關閉環境
    print("訓練完成")
    return episode_rewards

if __name__ == "__main__":
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="訓練 Pac-Man DQN 代理")
    parser.add_argument('--resume', action='store_true', help='從先前模型繼續訓練')
    parser.add_argument('--episodes', type=int, default=9000, help='訓練回合數')
    parser.add_argument('--early_stop_reward', type=float, default=4000, help='早期停止的獎勵閾值')
    parser.add_argument('--pretrain_episodes', type=int, default=100, help='預訓練的專家回合數')
    args = parser.parse_args()
    # 執行訓練
    train(resume=args.resume, episodes=args.episodes, early_stop_reward=args.early_stop_reward, 
          pretrain_episodes=args.pretrain_episodes)
    # 示例命令：
    # python .\ai\train.py --episodes=9000 --early_stop_reward=4000 --pretrain_episodes=100
    # 訓練新模型
    # python .\ai\train.py --resume --episodes=9000 --early_stop_reward=4000 --pretrain_episodes=100
    # 繼續訓練