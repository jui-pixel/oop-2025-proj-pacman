# ai/train.py
import os
import argparse
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler  # 導入 AMP
from environment import PacManEnv
from agent import DQNAgent
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
import random
import torch.nn.functional as F

def collect_expert_data(env, agent, num_episodes=100, max_steps_per_episode=200):
    """
    使用規則基礎 AI 收集專家數據，用於模仿學習預訓練，加入步數限制和多樣性。

    原理：
    - 專家數據由規則基礎 AI 生成，結合隨機探索增加多樣性。
    - 加入 max_steps_per_episode 限制，避免無限循環或低效數據。
    - 有效步驟過濾確保數據質量，減少無效轉換。

    Args:
        env (PacManEnv): Pac-Man 環境。
        agent (DQNAgent): DQN 代理。
        num_episodes (int): 收集專家數據的回合數。
        max_steps_per_episode (int): 每回合最大步數。

    Returns:
        list: 包含 (state, action) 的專家數據列表。
    """
    expert_data = []
    for episode in range(num_episodes):
        state, _ = env.reset(random_spawn_seed=episode)
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            # 90% 概率使用專家動作，10% 概率隨機動作增加多樣性
            if random.random() < 0.9:
                action = env.get_expert_action()
            else:
                action = random.randint(0, env.action_space.n - 1)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get('valid_step', False):
                expert_data.append((state, action))
                agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        print(f"專家回合 {episode + 1}/{num_episodes} 完成，步數: {steps}")
    return expert_data

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl",
          episodes=1000, early_stop_reward=2000, pretrain_episodes=100):
    """
    訓練 DQN 代理，結合模仿學習預訓練和早期停止機制。

    原理：
    - 訓練過程基於深度 Q 學習，結合模仿學習預訓練。
    - 使用 AMP 加速預訓練和主訓練過程。
    - ... (其他原理保持不變)

    Args:
        resume (bool): 是否從之前的模型繼續訓練。
        model_path (str): 模型保存/載入路徑。
        memory_path (str): 回放緩衝區保存/載入路徑。
        episodes (int): 訓練的總回合數。
        early_stop_reward (float): 平均獎勵閾值。
        pretrain_episodes (int): 模仿學習預訓練的回合數。

    Returns:
        list: 每個回合的總獎勵列表。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練設備：{device}")

    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device, batch_size=64)

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        agent.expert_prob = 0.01
        print(f"從 {model_path} 載入模型")

    if not resume:
        print(f"收集 {pretrain_episodes} 回合的專家數據進行預訓練...")
        expert_data = collect_expert_data(env, agent, pretrain_episodes, max_steps_per_episode=200)
        agent.pretrain(expert_data, pretrain_steps=10000)

    writer = SummaryWriter()
    episode_rewards = []
    recent_rewards = []

    state, _ = env.reset()
    for episode in range(episodes):
        total_reward = 0
        step_count = 0
        done = False
        state, _ = env.reset(random_spawn_seed=episode)
        agent.model.reset_noise()
        action_counts = np.zeros(action_dim)  # 動作分佈計數
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
            writer.add_scalar('Weight_Sigma_Mean', noise_metrics['weight_sigma_mean'], agent.steps)
            writer.add_scalar('Bias_Sigma_Mean', noise_metrics['bias_sigma_mean'], agent.steps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get('valid_step', False):
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn(expert_action=expert_action)
                if loss is not None:
                    writer.add_scalar('Loss', loss, agent.steps)
                total_reward += reward
                step_count += 1
            state = next_state
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        # 記錄額外指標
        writer.add_scalar('Mean_Q_Value', np.mean(q_values_list) if q_values_list else 0, episode)
        for i in range(action_dim):
            writer.add_scalar(f'Action_{i}_Ratio', action_counts[i] / max(step_count, 1), episode)
        if (episode + 1) % 1 == 0:
            print(f"回合 {episode+1}/{episodes}，獎勵：{total_reward:.2f}，"
                  f"步數：{step_count}，專家概率：{agent.expert_prob:.3f}，"
                  f"平均 Q 值：{np.mean(q_values_list):.4f}")

        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Expert Probability', agent.expert_prob, episode)

        if (episode + 1) % 10 == 0:
            agent.save(model_path, memory_path)
            print(f"在回合 {episode+1} 保存模型")

        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= early_stop_reward:
            print(f"早期停止：最近 100 回合平均獎勵 {np.mean(recent_rewards[-100:]):.2f} >= {early_stop_reward}")
            break

    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

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
    args = parser.parse_args()
    train(resume=args.resume, episodes=args.episodes, early_stop_reward=args.early_stop_reward,
          pretrain_episodes=args.pretrain_episodes)