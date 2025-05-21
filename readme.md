# Pac-Man AI 專案

這是一個基於 Pygame 的 Pac-Man 遊戲實現，結合了深度強化學習（DQN）和規則基礎的 AI 控制策略。專案支援隨機迷宮生成、多樣化的鬼魂行為、玩家控制以及 AI 自動控制，同時提供訓練和可視化工具來分析 DQN 代理的學習過程。

## 📂 專案結構

以下是專案的目錄結構及各模組的說明：

```
pacman_ai/
├── main.py                 # 遊戲主程式，負責初始化和運行遊戲
├── game/                   # 遊戲邏輯與環境模組
│   ├── __init__.py
│   ├── game.py            # 遊戲核心邏輯，管理狀態更新和碰撞檢測
│   ├── environment.py     # 強化學習環境，符合 OpenAI Gym 規範
│   ├── maze_generator.py  # 隨機迷宮生成器
│   ├── entities.py        # 定義遊戲實體（Pac-Man、鬼魂、能量球等）
│   └── settings.py        # 遊戲設定（尺寸、速度、分數等）
├── ghosts/                 # 鬼魂行為模組
│   ├── __init__.py
│   ├── basic_ghost.py     # 基礎鬼魂類別，提供通用行為
│   ├── ghost1.py          # 鬼魂 1：直接追逐 Pac-Man
│   ├── ghost2.py          # 鬼魂 2：預測 Pac-Man 前進方向
│   ├── ghost3.py          # 鬼魂 3：與 Ghost1 協同追擊
│   └── ghost4.py          # 鬼魂 4：靠近時隨機移動，遠離時追逐
├── ai/                     # 人工智慧模組
│   ├── __init__.py
│   ├── agent.py           # DQN 代理，管理記憶和訓練邏輯
│   ├── dqn.py             # DQN 神經網路模型定義
│   ├── strategies.py      # 控制策略（玩家、規則 AI、DQN AI）
│   ├── train.py           # DQN 訓練迴圈
│   └── test_cuda.py       # 檢查 CUDA 環境的工具腳本
├── assets/                 # 遊戲資源（音效、圖像等）
├── plot_metrics.py         # 繪製訓練獎勵和損失圖表的腳本
├── config.py               # 遊戲常量和配置
├── .gitignore              # Git 忽略檔案
├── readme.md               # 專案說明文件
└── classdiagram.md         # 類別圖，描述物件關係
```

## 🛠️ 模組詳細說明

### **1. 主程式**

- `main.py`\
  遊戲入口，初始化 Pygame，設置遊戲視窗，處理事件並渲染畫面。整合 `ControlManager` 來切換玩家控制和 AI 控制模式。

### **2. 遊戲邏輯模組 (**`game/`**)**

- `game.py`\
  實現遊戲核心邏輯，包括初始化實體、更新狀態和碰撞檢測。管理 Pac-Man、鬼魂、能量球和分數球的互動。
- `environment.py`\
  提供符合 OpenAI Gym 規範的強化學習環境，支援 DQN 訓練。定義狀態、動作空間和獎勵機制。
- `maze_generator.py`\
  隨機生成對稱的迷宮，包含牆壁、路徑、能量球和鬼魂重生點。使用牆壁擴展和路徑縮窄演算法確保連通性。
- `entities.py`\
  定義遊戲實體（Pac-Man、鬼魂、能量球、分數球）的行為，包括移動、吃球和規則基礎的 AI 邏輯。
- `settings.py`\
  配置遊戲參數，如迷宮尺寸、角色速度、分數規則和顏色常量。

### **3. 鬼魂行為模組 (**`ghosts/`**)**

- `basic_ghost.py`\
  抽象鬼魂類別，提供通用功能（如 BFS 路徑尋找、隨機移動），子類可覆寫追逐邏輯。
- `ghost1.py`\
  使用 BFS 直接追逐 Pac-Man，若無路徑則隨機移動到附近目標。
- `ghost2.py`\
  預測 Pac-Man 的移動方向，瞄準其前方 4 格。
- `ghost3.py`\
  與 Ghost1 協同，瞄準 Pac-Man 前方 2 格與 Ghost1 的對稱點。
- `ghost4.py`\
  當與 Pac-Man 距離小於 8 格時隨機移動，否則追逐。

### **4. 人工智慧模組 (**`ai/`**)**

- `agent.py`\
  實現 DQN 代理，管理記憶緩衝區、動作選擇和模型訓練。支援模型保存和載入。
- `dqn.py`\
  定義 DQN 神經網路，使用卷積層和全連接層預測動作 Q 值。
- `strategies.py`\
  定義控制策略，包括玩家控制、規則基礎 AI 和 DQN AI。支援模式切換（按 'a' 鍵）。
- `train.py`\
  執行 DQN 訓練迴圈，記錄獎勵和損失到 TensorBoard 和 JSON 檔案。
- `test_cuda.py`\
  檢查 CUDA 是否可用，顯示 GPU 資訊，幫助驗證環境設置。

### **5. 可視化工具**

- `plot_metrics.py`\
  從 TensorBoard 日誌或 JSON 檔案提取訓練數據，繪製獎勵和損失圖表，保存為 PNG。

### **6. 配置與資源**

- `config.py`\
  定義遊戲常量，如迷宮尺寸、顏色、分數規則和幀率。
- `assets/`\
  存放遊戲資源，如音效和圖像素材（目前為空，需自行添加）。

### **7. 其他檔案**

- `.gitignore`\
  忽略 Python 快取檔案（`__pycache__`、`.pyc` 等）。
- `readme.md`\
  本文件，提供專案概述和模組說明。
- `classdiagram.md`\
  使用 Mermaid 描述類別關係，展示遊戲實體和核心物件的結構。

## 🚀 快速開始

### **環境要求**

- Python 3.8+
- Pygame
- PyTorch（支援 CUDA 可選）
- Matplotlib
- TensorBoard
- NumPy

安裝依賴：

```bash
pip install pygame torch numpy matplotlib tensorboard
```

### **運行遊戲**

執行以下命令啟動遊戲：

```bash
python main.py
```

- **控制方式**：
  - 使用方向鍵（↑↓←→）控制 Pac-Man。
  - 按 'a' 鍵在玩家控制、規則基礎 AI 和 DQN AI 之間切換。
  - 按關閉視窗退出遊戲。

### **訓練 DQN 代理**

執行以下命令開始訓練：

```bash
python ai/train.py
```

- 訓練結果（獎勵和損失）將記錄到 `runs/` 目錄和 `episode_rewards.json`。
- 模型和記憶緩衝區將保存為 `pacman_dqn_final.pth` 和 `replay_buffer_final.pkl`。

### **可視化訓練結果**

執行以下命令繪製訓練圖表：

```bash
python plot_metrics.py
```

- 圖表將保存為 `training_metrics.png`。

### **檢查 CUDA 環境**

執行以下命令驗證 CUDA 設置：

```bash
python ai/test_cuda.py
```

## 💡 作者

此專案由 Group 6 開發，感謝所有成員的貢獻！

## 📜 許可證

本專案採用 MIT 許可證，詳情見 `LICENSE` 檔案（需自行添加）。設定（尺寸、速度、分數等）