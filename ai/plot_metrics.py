# ai/plot_metrics.py
"""
繪製 DQN 訓練過程中的獎勵和損失圖表，從 TensorBoard 日誌或 JSON 檔案提取數據。
使用 Matplotlib 生成圖表並保存為 PNG 檔案，支援離線分析。

這個檔案的目的是可視化訓練過程中的關鍵指標（獎勵和損失），幫助分析模型的學習效果。
"""

# 導入必要的 Python 模組
import os  # 用於處理檔案路徑和目錄操作
import json  # 用於讀取 JSON 格式的獎勵數據
import matplotlib.pyplot as plt  # 用於繪製圖表
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # 用於讀取 TensorBoard 日誌
import numpy as np  # 用於數值計算和陣列操作

def extract_metrics_from_runs(runs_dir="runs"):
    """
    從 TensorBoard 的 runs 目錄中提取獎勵和損失數據。

    這個函數遍歷 TensorBoard 日誌目錄，提取標籤為 'Reward' 和 'Loss' 的數據，並按步數排序。

    Args:
        runs_dir (str): TensorBoard 日誌目錄，預設為 "runs"。

    Returns:
        tuple: 包含兩個列表的元組：
            - rewards: 獎勵數據，格式為 [(回合編號, 獎勵值), ...]
            - losses: 損失數據，格式為 [(訓練步數, 損失值), ...]
    """
    # 初始化儲存獎勵和損失的列表
    rewards = []  # 儲存獎勵數據
    losses = []   # 儲存損失數據

    # 遍歷 runs 目錄中的所有子目錄
    for subdir in os.listdir(runs_dir):
        # 構建子目錄的完整路徑
        log_path = os.path.join(runs_dir, subdir)
        # 如果不是目錄，則跳過
        if not os.path.isdir(log_path):
            continue
        # 創建 EventAccumulator 物件，用於讀取 TensorBoard 日誌
        event_acc = EventAccumulator(log_path)
        # 刷新日誌數據，確保載入最新內容
        event_acc.Reload()

        # 提取獎勵數據（標籤為 'Reward'）
        if 'Reward' in event_acc.Tags()['scalars']:
            # 從日誌中獲取獎勵數據
            reward_scalars = event_acc.Scalars('Reward')
            # 將數據轉為 (回合編號, 獎勵值) 格式，回合編號從 1 開始
            rewards.extend([(s.step + 1, s.value) for s in reward_scalars])

        # 提取損失數據（標籤為 'Loss'）
        if 'Loss' in event_acc.Tags()['scalars']:
            # 從日誌中獲取損失數據
            loss_scalars = event_acc.Scalars('Loss')
            # 將數據轉為 (訓練步數, 損失值) 格式
            losses.extend([(s.step, s.value) for s in loss_scalars])

    # 按步數排序獎勵和損失數據，確保順序正確
    rewards = sorted(rewards, key=lambda x: x[0])
    losses = sorted(losses, key=lambda x: x[0])
    # 返回獎勵和損失數據
    return rewards, losses

def load_rewards_from_json(json_path="episode_rewards.json"):
    """
    從 JSON 檔案載入獎勵數據。

    這個函數讀取儲存在 JSON 檔案中的回合獎勵數據，並轉換為 (回合編號, 獎勵值) 格式。

    Args:
        json_path (str): JSON 檔案路徑，預設為 "episode_rewards.json"。

    Returns:
        list: 回合編號和對應獎勵的列表，格式為 [(回合編號, 獎勵值), ...]。
    """
    # 檢查 JSON 檔案是否存在
    if os.path.exists(json_path):
        # 開啟並讀取 JSON 檔案
        with open(json_path, 'r') as f:
            rewards = json.load(f)  # 載入獎勵數據
        # 將獎勵數據轉為 (回合編號, 獎勵值) 格式，回合編號從 1 開始
        return [(i + 1, r) for i, r in enumerate(rewards)]
    # 如果檔案不存在，返回空列表
    return []

def plot_metrics(runs_dir="runs", json_path="episode_rewards.json"):
    """
    繪製訓練過程中的獎勵和損失圖表，優先從 JSON 檔案載入獎勵，否則從 TensorBoard 提取。

    這個函數生成兩個子圖：一個顯示回合獎勵，另一個顯示訓練損失，並將圖表保存為 PNG 檔案。

    Args:
        runs_dir (str): TensorBoard 日誌目錄，預設為 "runs"。
        json_path (str): 獎勵 JSON 檔案路徑，預設為 "episode_rewards.json"。
    """
    # 優先嘗試從 JSON 檔案載入獎勵數據
    rewards = load_rewards_from_json(json_path)
    # 如果 JSON 檔案沒有數據，則從 TensorBoard 日誌提取獎勵和損失
    if not rewards:
        rewards, losses = extract_metrics_from_runs(runs_dir)
    else:
        # 否則只從 TensorBoard 提取損失數據
        _, losses = extract_metrics_from_runs(runs_dir)

    # 檢查是否有任何數據可用
    if not rewards and not losses:
        print("No data found in runs or episode_rewards.json")
        return

    # 創建一個包含兩個子圖的畫布，尺寸為 10x8 英寸
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 繪製獎勵圖
    if rewards:
        # 解包回合編號和獎勵值
        episodes, reward_values = zip(*rewards)
        # 繪製折線圖，顏色為藍色
        ax1.plot(episodes, reward_values, label='Reward', color='blue')
        # 設定 X 軸標籤為 "Episode"（回合）
        ax1.set_xlabel('Episode')
        # 設定 Y 軸標籤為 "Total Reward"（總獎勵）
        ax1.set_ylabel('Total Reward')
        # 設定圖表標題為 "Training Rewards"（訓練獎勵）
        ax1.set_title('Training Rewards')
        # 添加網格線，提升可讀性
        ax1.grid(True)
        # 添加圖例
        ax1.legend()

    # 繪製損失圖
    if losses:
        # 解包訓練步數和損失值
        steps, loss_values = zip(*losses)
        # 繪製折線圖，顏色為紅色
        ax2.plot(steps, loss_values, label='Loss', color='red')
        # 設定 X 軸標籤為 "Training Step"（訓練步數）
        ax2.set_xlabel('Training Step')
        # 設定 Y 軸標籤為 "Loss"（損失）
        ax2.set_ylabel('Loss')
        # 設定圖表標題為 "Training Loss"（訓練損失）
        ax2.set_title('Training Loss')
        # 添加網格線，提升可讀性
        ax2.grid(True)
        # 添加圖例
        ax2.legend()

    # 自動調整子圖間距，防止標籤重疊
    plt.tight_layout()
    # 將圖表保存為 PNG 檔案
    plt.savefig("training_metrics.png")
    
if __name__ == "__main__":
    # 如果直接執行此檔案，則調用繪圖函數
    plot_metrics()
    # 印出完成提示
    print("Metrics plotted and saved as 'training_metrics.png'.")