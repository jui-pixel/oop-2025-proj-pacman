# Pac-Man AI 專案

這是一個基於 Pygame 的 Pac-Man 遊戲實現，結合深度 Q 學習（DQN）與規則基礎的 AI 控制策略。專案支援隨機迷宮生成、多樣化的鬼魂行為、玩家控制以及 AI 自動控制，並提供 DQN 代理訓練與訓練結果可視化工具。

## 📂 專案結構

以下是專案的目錄結構與主要模組說明：

```
oop-2025-proj-pacman/
├── main.py                 # 遊戲主程式，負責初始化與運行遊戲
├── config.py               # 遊戲常量與配置（迷宮尺寸、顏色等）
├── scores.json             # 儲存遊戲分數（可選）
├── .gitattributes          # Git 屬性配置
├── .gitignore              # Git 忽略檔案（Python 快取等）
├── ai/                     # AI 與 DQN 相關模組
│   ├── agent.py           # DQN 代理，管理記憶緩衝與訓練邏輯
│   ├── dqn.py             # DQN 神經網路模型定義
│   ├── environment.py     # 舊版 RL 環境（建議使用 game/environment.py）
│   ├── plot_metrics.py    # 繪製訓練獎勵與損失圖表，輸出為 PNG
│   ├── sumtree.py         # 優先經驗回放的 SumTree 結構
│   ├── test_cuda.py       # 檢查 CUDA 可用性的工具腳本
│   ├── train.py           # DQN 訓練迴圈，支援 TensorBoard 記錄
│   ├── __init__.py
│   └── __pycache__/
├── game/                   # 遊戲邏輯與環境模組
│   ├── game.py            # 核心遊戲邏輯，管理狀態更新與碰撞檢測
│   ├── environment.py     # 符合 Gym 規範的 RL 環境，支援可切換渲染
│   ├── maze_generator.py  # 隨機迷宮生成器，包含牆壁與路徑
│   ├── menu.py            # 遊戲選單（若已實現）
│   ├── renderer.py        # 使用 Pygame 渲染遊戲畫面
│   ├── strategies.py      # 控制策略（玩家、規則 AI、DQN AI）
│   ├── __init__.py
│   ├── entities/          # 遊戲實體定義
│   │   ├── entity_base.py # 實體基類
│   │   ├── entity_initializer.py # 初始化遊戲實體
│   │   ├── ghost.py       # 鬼魂實體，包含規則基礎行為
│   │   ├── pacman.py      # Pac-Man 實體，包含移動邏輯
│   │   ├── pellets.py     # 分數球與能量球
│   │   ├── __init__.py
│   │   └── __pycache__/
│   └── __pycache__/
├── docx/                   # 文件資料夾
│   ├── classdiagram.md    # 類別圖，描述物件關係
│   ├── requirements.txt   # Python 依賴清單
├── runs/                   # TensorBoard 訓練記錄
├── members/                # 貢獻者資訊
│   ├── 113511002.txt
│   ├── 113511034.txt
│   ├── 113511264.txt
└── __pycache__/
```

## 🛠️ 模組詳細說明

### **1. 主程式**
- `main.py`  
  遊戲入口，初始化 Pygame，設置遊戲視窗，處理事件並渲染畫面。透過 `strategies.py` 切換玩家控制、規則基礎 AI 與 DQN AI 模式。

### **2. 遊戲邏輯模組 (`game/`)**
- `game.py`  
  實現核心遊戲邏輯，包括實體初始化、狀態更新與碰撞檢測，管理 Pac-Man、鬼魂與球的互動。
- `environment.py`  
  提供符合 OpenAI Gym 規範的強化學習環境，支援 DQN 訓練。包含可切換渲染（透過 `--visualize` 啟用）、6 通道狀態表示以及獎勵機制（吃球、吃鬼、遊戲結束）。
- `maze_generator.py`  
  隨機生成對稱迷宮，包含牆壁、路徑、能量球與鬼魂重生點，使用牆壁擴展演算法確保連通性。
- `renderer.py`  
  使用 Pygame 渲染遊戲狀態，顯示 Pac-Man、鬼魂、球與迷宮。
- `strategies.py`  
  定義控制策略：玩家控制（方向鍵）、規則基礎 AI 與 DQN AI，支援透過 'a' 鍵切換模式。
- `entities/`  
  - `pacman.py`：實現 Pac-Man 的移動與碰撞邏輯。
  - `ghost.py`：定義鬼魂行為（追逐、逃跑、隨機移動）。
  - `pellets.py`：管理分數球與能量球。
  - `entity_initializer.py`：設置實體初始位置。
  - `entity_base.py`：實體基類。

### **3. AI 模組 (`ai/`)**
- `agent.py`  
  實現 DQN 代理，包含優先經驗回放、增強的 epsilon-greedy 探索（線性衰減與預熱）以及模型儲存/載入。
- `dqn.py`  
  定義 Dueling DQN 神經網路，使用卷積層與全連接層預測動作 Q 值。
- `train.py`  
  執行 DQN 訓練迴圈，將獎勵與損失記錄至 TensorBoard（`runs/`）與 `episode_rewards.json`，並儲存模型。
- `sumtree.py`  
  支援 DQN 代理的優先經驗回放。
- `plot_metrics.py`  
  繪製訓練獎勵與損失曲線，儲存為 `training_metrics.png`。
- `test_cuda.py`  
  檢查 CUDA 可用性並顯示 GPU 資訊。
- `environment.py`  
  舊版 RL 環境，建議使用 `game/environment.py` 進行訓練。

### **4. 配置與文件**
- `config.py`  
  定義遊戲常量，例如迷宮尺寸、顏色、FPS 與計分規則。
- `docx/classdiagram.md`  
  使用 Mermaid 語法描述類別關係。
- `docx/requirements.txt`  
  列出 Python 依賴。
- `scores.json`  
  儲存遊戲分數（可選，可能未使用）。

## 🚀 快速開始

### **環境要求**
- Python 3.13.2（或 3.8 以上）
- Pygame 2.6.1
- PyTorch（支援 CUDA 可選）
- NumPy
- Matplotlib
- TensorBoard

安裝依賴：
```bash
pip install -r docx/requirements.txt
```

或手動安裝：
```bash
pip install pygame torch numpy matplotlib tensorboard
```

### **運行遊戲**
啟動遊戲：
```bash
python main.py
```

- **控制方式**：
  - 方向鍵（↑↓←→）：控制 Pac-Man 移動。
  - 'a' 鍵：在玩家控制、規則基礎 AI 與 DQN AI 模式間切換。
  - 關閉視窗退出遊戲。

### **訓練 DQN 代理**
開始訓練：
```bash
python ai/train.py --episodes 1000 [--visualize] [--resume] [--render_frequency 10]
```

- 選項：
  - `--visualize`：啟用訓練過程中的 Pygame 渲染。
  - `--resume`：載入儲存的模型與記憶緩衝。
  - `--episodes`：訓練回合數（預設：1000）。
  - `--render_frequency`：每 N 步渲染一次（預設：10）。
- 輸出：
  - TensorBoard 記錄儲存於 `runs/`。
  - 獎勵數據儲存於 `episode_rewards.json`。
  - 模型與記憶緩衝儲存為 `pacman_dqn_final.pth` 與 `replay_buffer_final.pkl`。

### **可視化訓練結果**
生成訓練圖表：
```bash
python ai/plot_metrics.py
```

- 將獎勵與損失曲線儲存為 `training_metrics.png`。

### **檢查 CUDA 環境**
驗證 CUDA 設置：
```bash
python ai/test_cuda.py
```

## 🐛 已知問題與修復
- **遊戲結束錯誤**：修復了 Pac-Man 被鬼魂吃掉後回合未正確重啟的問題，透過追蹤生命數與確保重置邏輯實現。
- **渲染**：新增可切換渲染功能，透過 `--visualize` 控制，禁用時跳過 Pygame 初始化，提升效能。
- **Epsilon 探索**：增強 DQN 的 epsilon-greedy 探索，加入線性衰減與預熱，改善訓練效果。

## 💡 作者
由 Group 6 開發，感謝 `members/` 中列出的所有貢獻者！

## 📜 許可證
本專案採用 MIT 許可證，詳情見 `LICENSE` 檔案（需自行添加）。