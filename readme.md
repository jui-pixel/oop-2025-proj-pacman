# Pac-Man

這是一個基於 Pygame 的 Pac-Man 遊戲實現，採用物件導向程式設計（OOP）原則，結合深度 Q 學習（DQN）與規則基礎 AI 控制策略。專案支援隨機迷宮生成、多樣化的鬼魂行為、玩家控制與 AI 自動控制，並提供 DQN 代理訓練與訓練結果可視化工具。OOP 的應用確保了程式碼的模組化、可維護性和可擴展性，透過類別設計實現了遊戲邏輯的清晰分層。

## 📂 專案結構

以下是專案的目錄結構與主要模組說明：

```
oop-2025-proj-pacman/
├── main.py                 # 遊戲主程式，負責初始化與運行遊戲
├── config.py               # 遊戲常量與配置（迷宮尺寸、顏色等）
├── scores.json             # 儲存遊戲分數
├── .gitattributes          # Git 屬性配置
├── .gitignore              # Git 忽略檔案
├── ai/                     # AI 與 DQN 相關模組
│   ├── agent.py           # DQN 代理，管理記憶緩衝與訓練邏輯
│   ├── dqn.py             # DQN 神經網路模型定義
│   ├── environment.py     # RL 環境
│   ├── plot_metrics.py    # 繪製訓練獎勵與損失圖表，輸出為 PNG
│   ├── sumtree.py         # 優先經驗回放的 SumTree 結構
│   ├── test_cuda.py       # 檢查 CUDA 可用性的工具腳本
│   ├── train.py           # DQN 訓練迴圈，支援 TensorBoard 記錄
│   ├── __init__.py
│   └── __pycache__/
├── game/                   # 遊戲邏輯與環境模組
│   ├── game.py            # 核心遊戲邏輯，管理狀態更新與碰撞檢測
│   ├── maze_generator.py  # 隨機迷宮生成器，包含牆壁與路徑
│   ├── menu.py            # 遊戲選單
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
├── members/
│   ├── 113511002.txt
│   ├── 113511034.txt
│   ├── 113511264.txt
└── __pycache__/
```

### **配置與文件**
- **`config.py`**：定義常量（如 `MAZE_WIDTH`、`CELL_SIZE`），作為全局配置，減少硬編碼。
- **`docx/classdiagram.md`**：使用 Mermaid 描述類別關係，展示 OOP 結構。
- **`docx/requirements.txt`**：列出依賴，確保環境一致性。

## 🚀 快速開始

### **環境要求**
- Python 3.13.2（或 3.8 以上）
- Pygame 2.6.1
- PyTorch（支援 CUDA 可選，DQN AI 必須）
- NumPy
- Matplotlib
- TensorBoard
- Optuna

安裝依賴：
```bash
pip install -r docx/requirements.txt
```

或手動安裝：
```bash
pip install pygame torch numpy matplotlib tensorboard optuna
```

### **運行遊戲**
啟動遊戲：
```bash
python main.py
```

- **控制方式**：
  - 方向鍵（↑↓←→）：控制 Pac-Man 移動。
  - ESC：暫停遊戲。

### 訓練 DQN 代理

執行 `train.py` 以訓練 Pac-Man DQN 代理，支援模仿學習預訓練、混合精度訓練（AMP）和早期停止機制。

#### 執行命令
```bash
python .\ai\train.py 
python .\ai\train.py --resume
python .\ai\train.py --optuna
```

## 選項

### 通用選項
- **`--resume`**（布林值，預設：`False`）  
  從先前保存的模型和回放緩衝區繼續訓練。  
  - 使用 `--model_path` 和 `--memory_path` 指定的路徑。
- **`--optuna`**（布林值，預設：`False`）  
  啟用 Optuna 進行超參數優化，自動搜索最佳參數組合。  
  - 執行 50 次試驗，儲存結果於 `sqlite:///optuna.db`。

### 訓練設置
- **`--episodes`**（整數，預設：`1000`）  
  總訓練回合數。
- **`--pretrain_episodes`**（整數，預設：`100`）  
  預訓練時收集專家數據的回合數。
- **`--early_stop_reward`**（浮點數，預設：`10000`）  
  當最近 100 回合平均獎勵達到此閾值時，提前停止訓練。
- **`--model_path`**（字串，預設：`"pacman_dqn.pth"`）  
  模型保存/載入的檔案路徑。
- **`--memory_path`**（字串，預設：`"replay_buffer.pkl"`）  
  回放緩衝區保存/載入的檔案路徑。

### DQN 模型參數
- **`--lr`**（浮點數，預設：`0.001`）  
  學習率，控制模型參數更新的步長。
- **`--batch_size`**（整數，預設：`64`）  
  每次訓練的批量大小。
- **`--target_update_freq`**（整數，預設：`10`）  
  目標網絡更新的回合頻率。
- **`--sigma`**（浮點數，預設：`0.5`）  
  Noisy DQN 層的噪聲因子，用於探索。
- **`--n_step`**（整數，預設：`8`）  
  n 步回報的步數，影響長期獎勵計算。
- **`--gamma`**（浮點數，預設：`0.95`）  
  折扣因子，控制未來獎勵的權重。
- **`--alpha`**（浮點數，預設：`0.8`）  
  優先級經驗回放的 alpha 參數，控制採樣優先級。
- **`--beta`**（浮點數，預設：`0.6`）  
  優先級經驗回放的 beta 參數，控制重要性採樣權重。
- **`--beta_increment`**（浮點數，預設：`0.0001`）  
  每步訓練的 beta 增量。
- **`--expert_prob_start`**（浮點數，預設：`0.3`）  
  專家行動策略的初始概率。
- **`--expert_prob_end`**（浮點數，預設：`0.01`）  
  專家行動策略的結束概率。
- **`--expert_prob_decay_steps`**（整數，預設：`500000`）  
  專家概率從起始到結束的衰減步數。
- **`--ghost_penalty_weight`**（浮點數，預設：`3.0`）  
  靠近鬼魂時的懲罰權重，影響獎勵計算。

### 專家數據收集參數
- **`--expert_episodes`**（整數，預設：`100`）  
  收集專家數據的回合數。
- **`--expert_max_steps_per_episode`**（整數，預設：`200`）  
  每個專家回合的最大步數。
- **`--expert_random_prob`**（浮點數，預設：`0.1`）  
  專家數據收集時隨機行動的概率。
- **`--max_expert_data`**（整數，預設：`10000`）  
  收集的最大專家數據量。
  
#### 輸出

- **TensorBoard 記錄**：儲存於 `runs/`，包含以下指標：
  - `Reward`：每回合總獎勵。
  - `Mean_Q_Value`：平均 Q 值。
  - `Loss`：訓練損失。
  - `Lives_Lost`：每回合生命損失次數。
  - `Action_i_Ratio`：動作分佈比例（上、下、左、右）。
  - `Expert_Probability`：當前專家概率。
  - NoisyLinear 噪聲指標（例如 `FC1_Weight_Sigma_Mean`）。
- **模型與記憶緩衝**：儲存為 `pacman_dqn_final.pth` 和 `replay_buffer_final.pkl`。
- **回合數據**：每回合的獎勵儲存於 `episode_rewards.json`，生命損失儲存於 `episode_lives_lost.json`。


### **可視化訓練結果**
生成訓練圖表：
```bash
python ai/plot_metrics.py
```

- 將獎勵與損失曲線儲存為 `training_metrics.png`。

在tensorboard中查看(可用於訓練中)
```bash
tensorboard --logdir runs
```

### **檢查 CUDA 環境**
```bash
python ai/test_cuda.py
```

## 🐛 已知問題
- **DQN AI** : 會主動尋找能量球，小機率吃鬼，但躲避鬼魂的能力不足。
- **預計改進**：
  - 增強選單功能（排行榜篩選、設定自訂選項）。
  - 添加動畫效果，改善視覺體驗。
  - 優化 PacmanEnv 的獎勵機制。(加強躲鬼策略試驗中)

## 💡 作者
由 Group 6 開發，感謝 `members/` 中列出的貢獻者。

## 參考文獻
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://dl.acm.org/doi/10.1145/3292500.3330701)
    - 超參數優化器 
- [Pac-Man Maze Generation](https://shaunlebron.github.io/pacman-mazegen/)
    - 迷宮生成設計靈感來源
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236?wm=book_wap_0005)
    - 基礎 DQN 架構
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
    - 加強DQN
- [PyTorch Offical Website](https://pytorch.org/)
    - 訓練DQN
- [PyTorch: An imperative style, high-performance deep learning library](https://arxiv.org/abs/1912.01703)
    - 訓練DQN
- [OpenAI Gym](https://github.com/openai/gym)
    - 環境建構靈感來源
- [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
    - 模仿學習結合強化學習的靈感來源

