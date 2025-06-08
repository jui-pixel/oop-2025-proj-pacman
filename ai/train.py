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

def train(resume=False, model_path="pacman_dqn.pth", memory_path="replay_buffer.pkl", 
          episodes=1000, early_stop_reward=2000):
    """
    訓練 DQN 代理，使用單一環境並支援早期停止。

    Args:
        resume (bool): 是否從之前的模型繼續訓練。
        model_path (str): 模型保存/載入路徑。
        memory_path (str): 回放緩衝區保存/載入路徑。
        episodes (int): 訓練的總回合數。
        early_stop_reward (float): 平均獎勵閾值，若達到則提前停止。

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
            action = agent.choose_action(state)  # 選擇動作
            next_state, reward, terminated, truncated, info = env.step(action)  # 執行動作
            done = terminated or truncated
            # 僅在移動完成時儲存和學習
            if info['valid_step'] == True:
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn()  # 學習更新
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
                  f"Steps: {step_count}, Epsilon: {agent.epsilon:.3f}")

        # 記錄到 TensorBoard
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)

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
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--early_stop_reward', type=float, default=2000, help='Reward threshold for early stopping')
    args = parser.parse_args()
    train(resume=args.resume, episodes=args.episodes, early_stop_reward=args.early_stop_reward)