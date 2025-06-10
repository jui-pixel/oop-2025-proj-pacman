# ai/train.py
import os
import argparse
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from environment import PacManEnv
from agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
import random
import torch.nn.functional as F
import torch.optim as optim

def collect_expert_data(env, agent, num_episodes=100, max_steps_per_episode=200, expert_random_prob=0.1, max_expert_data=10000):
    """
    使用規則基礎 AI 收集專家數據，用於模仿學習預訓練，加入步數限制和多樣性。

    原理：
    - 專家數據由規則基礎 AI 生成，結合隨機探索增加多樣性。
    - max_steps_per_episode 限制避免無限循環，expert_random_prob 控制隨機動作比例。
    - max_expert_data 限制總數據量，防止記憶體過載。

    Args:
        env (PacManEnv): Pac-Man 環境。
        agent (DQNAgent): DQN 代理。
        num_episodes (int): 收集專家數據的回合數。
        max_steps_per_episode (int): 每回合最大步數。
        expert_random_prob (float): 隨機動作的概率。
        max_expert_data (int): 最大專家數據量。

    Returns:
        list: 包含 (state, action) 的專家數據列表。
    """
    expert_data = []
    for episode in range(num_episodes):
        if len(expert_data) >= max_expert_data:
            break
        state, _ = env.reset(random_spawn_seed=episode)
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            if random.random() < expert_random_prob:
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = env.get_expert_action()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get('valid_step', False):
                expert_data.append((state, action))
                agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        print(f"專家回合 {episode + 1}/{num_episodes} 完成，步數: {steps}, 數據量: {len(expert_data)}")
    return expert_data[:max_expert_data]

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl",
          episodes=1000, early_stop_reward=2000, pretrain_episodes=100, lr=1e-3,
          batch_size=64, target_update_freq=10, sigma=0.5, n_step=8, gamma=0.95,
          expert_prob_start=0.3, expert_prob_end=0.01, expert_prob_decay_steps=500000,
          expert_random_prob=0.1, max_expert_data=10000, use_lr_scheduler=False):
    """
    訓練 DQN 代理，結合模仿學習預訓練和早期停止機制。

    原理：
    - 訓練過程基於深度 Q 學習，結合模仿學習預訓練。
    - 使用 AMP 加速預訓練和主訓練過程。
    - 支援學習率調度器、動態超參數調整和詳細日誌記錄。

    Args:
        resume (bool): 是否從之前的模型繼續訓練。
        model_path (str): 模型保存/載入路徑。
        memory_path (str): 回放緩衝區保存/載入路徑。
        episodes (int): 訓練的總回合數。
        early_stop_reward (float): 平均獎勵閾值。
        pretrain_episodes (int): 模仿學習預訓練的回合數。
        lr (float): 學習率。
        batch_size (int): 批量大小。
        target_update_freq (int): 目標網絡更新頻率。
        sigma (float): NoisyLinear 初始噪聲標準差。
        n_step (int): n-step 學習步數。
        gamma (float): 折扣因子。
        expert_prob_start (float): 初始專家概率。
        expert_prob_end (float): 最終專家概率。
        expert_prob_decay_steps (int): 專家概率衰減步數。
        expert_random_prob (float): 專家數據隨機動作概率。
        max_expert_data (int): 最大專家數據量。
        use_lr_scheduler (bool): 是否使用學習率調度器。

    Returns:
        list: 每個回合的總獎勵列表。
    """
    # 參數驗證
    if lr <= 0:
        raise ValueError("學習率 (lr) 必須大於 0")
    if batch_size <= 0:
        raise ValueError("批量大小 (batch_size) 必須大於 0")
    if target_update_freq <= 0:
        raise ValueError("目標網絡更新頻率 (target_update_freq) 必須大於 0")
    if sigma <= 0:
        raise ValueError("NoisyLinear 噪聲標準差 (sigma) 必須大於 0")
    if n_step <= 0:
        raise ValueError("n-step 學習步數 (n_step) 必須大於 0")
    if not 0 <= gamma < 1:
        raise ValueError("折扣因子 (gamma) 必須在 [0, 1) 範圍內")
    if not 0 <= expert_prob_start <= 1 or not 0 <= expert_prob_end <= 1:
        raise ValueError("專家概率 (expert_prob_start, expert_prob_end) 必須在 [0, 1] 範圍內")
    if expert_prob_decay_steps <= 0:
        raise ValueError("專家概率衰減步數 (expert_prob_decay_steps) 必須大於 0")
    if not 0 <= expert_random_prob <= 1:
        raise ValueError("專家數據隨機概率 (expert_random_prob) 必須在 [0, 1] 範圍內")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練設備：{device}")
    print(f"訓練參數：lr={lr}, batch_size={batch_size}, target_update_freq={target_update_freq}, "
          f"sigma={sigma}, n_step={n_step}, gamma={gamma}, expert_prob_start={expert_prob_start}, "
          f"expert_prob_end={expert_prob_end}, expert_prob_decay_steps={expert_prob_decay_steps}, "
          f"expert_random_prob={expert_random_prob}, max_expert_data={max_expert_data}, "
          f"use_lr_scheduler={use_lr_scheduler}")

    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        target_update_freq=target_update_freq,
        n_step=n_step,
        expert_prob_start=expert_prob_start,
        expert_prob_end=expert_prob_end,
        expert_prob_decay_steps=expert_prob_decay_steps
    )

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        agent.expert_prob = expert_prob_end
        print(f"從 {model_path} 載入模型")

    if not resume:
        print(f"收集 {pretrain_episodes} 回合的專家數據進行預訓練...")
        expert_data = collect_expert_data(
            env, agent, pretrain_episodes, max_steps_per_episode=200,
            expert_random_prob=expert_random_prob, max_expert_data=max_expert_data
        )
        agent.pretrain(expert_data, pretrain_steps=10000)

    writer = SummaryWriter()
    episode_rewards = []
    recent_rewards = []
    episode_lives_lost = []  # 新增：記錄每回合生命損失次數

    state, _ = env.reset()
    for episode in range(episodes):
        total_reward = 0
        step_count = 0
        lives_lost = 0
        done = False
        state, _ = env.reset(random_spawn_seed=episode)
        agent.model.reset_noise()
        action_counts = np.zeros(action_dim)
        q_values_list = []
        while not done:
            if random.random() < agent.expert_prob:
                action = env.get_expert_action()
                expert_action = True
            else:
                action = agent.choose_action(state)
                expert_action = False
            action_counts[action] += 1
            q_values, noise_metrics = agent.model(torch.FloatTensor(state).unsqueeze(0).to(device))
            q_values_list.append(q_values.detach().cpu().numpy().mean())
            # 記錄所有噪聲指標
            writer.add_scalar('FC1_Weight_Sigma_Mean', noise_metrics['fc1_weight_sigma_mean'], agent.steps)
            writer.add_scalar('FC1_Bias_Sigma_Mean', noise_metrics['fc1_bias_sigma_mean'], agent.steps)
            writer.add_scalar('Value_Weight_Sigma_Mean', noise_metrics['value_weight_sigma_mean'], agent.steps)
            writer.add_scalar('Value_Bias_Sigma_Mean', noise_metrics['value_bias_sigma_mean'], agent.steps)
            writer.add_scalar('Advantage_Weight_Sigma_Mean', noise_metrics['advantage_weight_sigma_mean'], agent.steps)
            writer.add_scalar('Advantage_Bias_Sigma_Mean', noise_metrics['advantage_bias_sigma_mean'], agent.steps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get('valid_step', False):
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn(expert_action=expert_action)
                if loss is not None:
                    writer.add_scalar('Loss', loss, agent.steps)
                total_reward += reward
                step_count += 1
                # 檢查生命損失
                if info.get('life_lost', False):
                    lives_lost += 1
            state = next_state
        episode_rewards.append(total_reward)
        episode_lives_lost.append(lives_lost)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        # 記錄額外指標
        writer.add_scalar('Mean_Q_Value', np.mean(q_values_list) if q_values_list else 0, episode)
        writer.add_scalar('Lives_Lost', lives_lost, episode)
        for i in range(action_dim):
            writer.add_scalar(f'Action_{i}_Ratio', action_counts[i] / max(step_count, 1), episode)
        if (episode + 1) % 1 == 0:
            print(f"回合 {episode+1}/{episodes}，獎勵：{total_reward:.2f}，"
                  f"步數：{step_count}，生命損失：{lives_lost}，"
                  f"專家概率：{agent.expert_prob:.3f}，平均 Q 值：{np.mean(q_values_list):.4f}")

        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Expert_Probability', agent.expert_prob, episode)

        if (episode + 1) % 10 == 0:
            agent.save(model_path, memory_path)
            print(f"在回合 {episode+1} 保存模型")

        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= early_stop_reward:
            print(f"早期停止：最近 100 回合平均獎勵 {np.mean(recent_rewards[-100:]):.2f} >= {early_stop_reward}")
            break

    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)
    with open("episode_lives_lost.json", "w") as f:
        json.dump(episode_lives_lost, f)

    writer.close()
    env.close()
    print("訓練完成")
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 Pac-Man DQN 代理")
    parser.add_argument('--resume', action='store_true', help='從先前模型繼續訓練')
    parser.add_argument('--episodes', type=int, default=5000, help='訓練回合數')
    parser.add_argument('--early_stop_reward', type=float, default=4000, help='早期停止的獎勵閾值')
    parser.add_argument('--pretrain_episodes', type=int, default=100, help='預訓練的專家回合數')
    parser.add_argument('--lr', type=float, default=1e-3, help='學習率')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--target_update_freq', type=int, default=10, help='目標網絡更新頻率')
    parser.add_argument('--sigma', type=float, default=0.5, help='NoisyLinear 初始噪聲標準差')
    parser.add_argument('--n_step', type=int, default=8, help='n-step 學習步數')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')
    parser.add_argument('--expert_prob_start', type=float, default=0.3, help='初始專家概率')
    parser.add_argument('--expert_prob_end', type=float, default=0.01, help='最終專家概率')
    parser.add_argument('--expert_prob_decay_steps', type=int, default=500000, help='專家概率衰減步數')
    parser.add_argument('--expert_random_prob', type=float, default=0.1, help='專家數據隨機動作概率')
    parser.add_argument('--max_expert_data', type=int, default=10000, help='最大專家數據量')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='是否使用學習率調度器')
    args = parser.parse_args()

    train(
        resume=args.resume,
        episodes=args.episodes,
        early_stop_reward=args.early_stop_reward,
        pretrain_episodes=args.pretrain_episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        sigma=args.sigma,
        n_step=args.n_step,
        gamma=args.gamma,
        expert_prob_start=args.expert_prob_start,
        expert_prob_end=args.expert_prob_end,
        expert_prob_decay_steps=args.expert_prob_decay_steps,
        expert_random_prob=args.expert_random_prob,
        max_expert_data=args.max_expert_data,
        use_lr_scheduler=args.use_lr_scheduler
    )