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
    使用規則基礎 AI 收集專家數據。

    Args:
        env (PacManEnv): Pac-Man 環境。
        agent (DQNAgent): DQN 代理，用於儲存數據。
        num_episodes (int): 收集數據的回合數。

    Returns:
        list: 包含 (state, action) 的專家數據列表。
    """
    expert_data = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.get_expert_action()  # 使用專家策略
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get('valid_step', False):
                expert_data.append((state, action))
                agent.store_transition(state, action, reward, next_state, done)
            state = next_state
        print(f"Expert episode {episode + 1}/{num_episodes} completed")
    return expert_data

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl", 
          episodes=1000, early_stop_reward=2000, pretrain_episodes=100):
    """
    訓練 DQN 代理，加入模仿學習預訓練，支援早期停止。

    Args:
        resume (bool): 是否從之前的模型繼續訓練。
        model_path (str): 模型保存/載入路徑。
        memory_path (str): 回放緩衝區保存/載入路徑。
        episodes (int): 訓練的總回合數。
        early_stop_reward (float): 平均獎勵閾值，若達到則提前停止。
        pretrain_episodes (int): 模仿學習預訓練的回合數。

    Returns:
        list: 每個回合的總獎勵。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 選擇設備
    print(f"Training on {device}")

    # 初始化環境
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED)
    state_dim = env.observation_space.shape  # 獲取狀態維度
    action_dim = env.action_space.n  # 獲取動作維度
    pacman = env.pacman
    # 初始化代理
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    if resume and os.path.exists(model_path):  # 若啟用繼續訓練且文件存在
        agent.load(model_path, memory_path)
        print(f"Loaded model from {model_path}")

    # 模仿學習預訓練
    if not resume:
        print(f"Collecting {pretrain_episodes} episodes of expert data for pretraining...")
        expert_data = collect_expert_data(env, agent, pretrain_episodes)
        agent.pretrain(expert_data, pretrain_steps=1000)  # 執行預訓練

    writer = SummaryWriter()  # 初始化 TensorBoard 記錄器
    episode_rewards = []  # 儲存每個回合的獎勵
    recent_rewards = []  # 儲存最近 100 回合的獎勵

    # 訓練循環
    state, _ = env.reset()
    for episode in range(episodes):
        total_reward = 0
        step_count = 0
        done = False
        state, _ = env.reset()  # 重置環境
        agent.model.reset_noise()
        while not done:
            # 結合專家策略和 DQN 策略
            if random.random() < agent.expert_prob:  # 使用專家策略的機率
                action = env.get_expert_action()
                expert_action = True
            else:
                action = agent.choose_action(state)
                expert_action = False
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # 僅在移動完成時儲存和學習
            if info.get('valid_step', False):
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn(expert_action=expert_action)
                if loss is not None:
                    writer.add_scalar('Loss', loss, agent.steps)  # 記錄損失
                total_reward += reward
                step_count += 1
            state = next_state

        # 記錄結果
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # 每回合打印進度
        if (episode + 1) % 1 == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, "
                  f"Steps: {step_count}, Epsilon: {agent.epsilon:.3f}, Expert Prob: {agent.expert_prob:.3f}")

        # 記錄到 TensorBoard
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('Expert Probability', agent.expert_prob, episode)

        # 定期保存模型
        if (episode + 1) % 10 == 0:
            agent.save(model_path, memory_path)
            print(f"Saved model at episode {episode+1}")

        # 檢查早期停止
        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= early_stop_reward:
            print(f"Early stopping: Avg reward {np.mean(recent_rewards[-100:]):.2f} >= {early_stop_reward}")
            break

    # 保存最終模型和獎勵
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)

    writer.close()  # 關閉 TensorBoard
    env.close()  # 關閉環境
    print("Training completed")
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man DQN Agent")
    parser.add_argument('--resume', action='store_true', help='Resume from previous model')
    parser.add_argument('--episodes', type=int, default=9000, help='Number of episodes')
    parser.add_argument('--early_stop_reward', type=float, default=4000, help='Reward threshold for early stopping')
    parser.add_argument('--pretrain_episodes', type=int, default=1000, help='Number of expert episodes for pretraining')
    args = parser.parse_args()
    train(resume=args.resume, episodes=args.episodes, early_stop_reward=args.early_stop_reward, 
          pretrain_episodes=args.pretrain_episodes)