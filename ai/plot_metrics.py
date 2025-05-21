# ai/plot_metrics.py
"""
繪製 DQN 訓練過程中的獎勵和損失圖表，從 TensorBoard 日誌或 JSON 檔案提取數據。
使用 Matplotlib 生成圖表並保存為 PNG 檔案。
"""
import os
import json
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def extract_metrics_from_runs(runs_dir="runs"):
    """
    從 TensorBoard 的 runs 目錄中提取獎勵和損失數據。

    Args:
        runs_dir (str): TensorBoard 日誌目錄，預設為 "runs"。

    Returns:
        Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]: 獎勵和損失數據，按步數排序。
    """
    rewards = []
    losses = []
    
    # 遍歷 runs 目錄中的所有子目錄
    for subdir in os.listdir(runs_dir):
        log_path = os.path.join(runs_dir, subdir)
        if not os.path.isdir(log_path):
            continue
        event_acc = EventAccumulator(log_path)
        event_acc.Reload()
        
        # 提取獎勵數據
        if 'Reward' in event_acc.Tags()['scalars']:
            reward_scalars = event_acc.Scalars('Reward')
            rewards.extend([(s.step + 1, s.value) for s in reward_scalars])  # 回合從 1 開始
        
        # 提取損失數據
        if 'Loss' in event_acc.Tags()['scalars']:
            loss_scalars = event_acc.Scalars('Loss')
            losses.extend([(s.step, s.value) for s in loss_scalars])
    
    # 按步數排序
    rewards = sorted(rewards, key=lambda x: x[0])
    losses = sorted(losses, key=lambda x: x[0])
    return rewards, losses

def load_rewards_from_json(json_path="episode_rewards.json"):
    """
    從 episode_rewards.json 檔案載入獎勵數據。

    Args:
        json_path (str): JSON 檔案路徑，預設為 "episode_rewards.json"。

    Returns:
        List[Tuple[int, float]]: 回合編號和對應獎勵的列表。
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            rewards = json.load(f)
        return [(i + 1, r) for i, r in enumerate(rewards)]
    return []

def plot_metrics(runs_dir="runs", json_path="episode_rewards.json"):
    """
    繪製訓練過程中的獎勵和損失圖表，優先從 JSON 檔案載入獎勵，否則從 runs 目錄提取。

    Args:
        runs_dir (str): TensorBoard 日誌目錄，預設為 "runs"。
        json_path (str): 獎勵 JSON 檔案路徑，預設為 "episode_rewards.json"。
    """
    # 嘗試從 JSON 載入獎勵，若失敗則從 runs 提取
    rewards = load_rewards_from_json(json_path)
    if not rewards:
        rewards, losses = extract_metrics_from_runs(runs_dir)
    else:
        _, losses = extract_metrics_from_runs(runs_dir)
    
    if not rewards and not losses:
        print("No data found in runs or episode_rewards.json")
        return
    
    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 繪製獎勵圖
    if rewards:
        episodes, reward_values = zip(*rewards)
        ax1.plot(episodes, reward_values, label='Reward', color='blue')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards')
        ax1.grid(True)
        ax1.legend()
    
    # 繪製損失圖
    if losses:
        steps, loss_values = zip(*losses)
        ax2.plot(steps, loss_values, label='Loss', color='red')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
        ax2.legend()
    
    # 調整佈局並保存圖表
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

if __name__ == "__main__":
    plot_metrics()