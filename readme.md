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

## 🛠️ OOP 設計與模組說明

本專案採用 OOP 原則（封裝、繼承、多型、抽象）設計，透過類別的層次結構實現模組化功能。以下是主要模組及其 OOP 應用的詳細說明：

### **1. 主程式 (`main.py`)**
- **功能**：遊戲入口，負責初始化 Pygame、設置視窗、運行主迴圈並處理事件。
- **OOP 應用**：
  - **封裝**：`main` 函數封裝了遊戲初始化與運行邏輯，透過依賴注入將 `Game`、`Renderer` 和 `ControlManager` 實例傳遞給相關模組。
  - **物件協作**：整合多個類別（`Game`、`Renderer`、`ControlManager`、`MenuButton`）進行協作，實現遊戲狀態管理、畫面渲染和控制切換。
  - **動態行為**：透過 `ControlManager` 的多型行為，支援玩家、規則 AI 和 DQN AI 的動態切換。

### **2. 遊戲邏輯模組 (`game/`)**

#### **`game.py`**
- **功能**：核心遊戲邏輯，管理狀態更新、碰撞檢測與實體互動。
- **類別**：`Game`
- **OOP 應用**：
  - **封裝**：將遊戲狀態（Pac-Man、鬼魂、迷宮、分數、生命值）封裝於 `Game` 類別，提供方法如 `update`、`get_pacman` 和 `get_final_score` 進行操作。
  - **物件管理**：透過組合（composition）持有 `PacMan`、`Ghost`、`PowerPellet` 和 `ScorePellet` 實例，統一管理實體更新與碰撞。
  - **狀態追蹤**：使用私有屬性（如 `_is_running`、`_death_animation_progress`）封裝遊戲狀態，僅透過公開方法訪問。

#### **`maze_generator.py`**
- **功能**：生成隨機對稱迷宮，包含牆壁、路徑、能量球等。
- **類別**：`Map`
- **OOP 應用**：
  - **封裝**：`Map` 類別封裝迷宮數據（格子陣列）與操作方法（如 `get_tile`、`set_tile`）。
  - **演算法實現**：內部實現牆壁擴展與路徑縮窄演算法，確保迷宮連通性，透過方法封裝生成邏輯。
  - **可擴展性**：提供公開接口，允許未來添加新迷宮生成策略。

#### **`menu.py`**
- **功能**：提供選單介面，包括主選單、排行榜、設定、暫停選單和遊戲結果。
- **類別**：`MenuButton`
- **OOP 應用**：
  - **封裝**：`MenuButton` 類別封裝按鈕的屬性（文字、位置、顏色）與行為（繪製、懸停檢測）。
  - **可重用性**：透過參數化建構函數，`MenuButton` 可在多個選單（主選單、暫停選單、結果畫面）中重用。
  - **行為分離**：選單邏輯分離為獨立函數（如 `show_menu`、`show_pause_menu`），與 `MenuButton` 類別協作，提升維護性。
  - **事件處理**：支援鍵盤與滑鼠事件，實現多型交互（鍵盤選擇與滑鼠點擊）。

#### **`renderer.py`**
- **功能**：負責遊戲畫面渲染，包括迷宮、Pac-Man、鬼魂和分數顯示。
- **類別**：`Renderer`
- **OOP 應用**：
  - **封裝**：`Renderer` 類別封裝 Pygame 畫面物件與渲染邏輯，透過 `render` 方法統一處理畫面更新。
  - **物件依賴**：依賴 `Game` 物件提供狀態數據（如迷宮、Pac-Man、鬼魂），實現邏輯與渲染的分離。
  - **動態效果**：透過條件渲染實現動畫效果（如 Pac-Man 死亡縮小、鬼魂閃爍），使用私有方法計算透明度與縮放比例。

#### **`strategies.py`**
- **功能**：實現控制策略，包括玩家控制、規則 AI 和 DQN AI。
- **類別**：`ControlStrategy`（抽象基類）、`PlayerControl`、`RuleBasedAIControl`、`DQNAIControl`、`ControlManager`
- **OOP 應用**：
  - **抽象與繼承**：`ControlStrategy` 定義抽象方法 `move`，透過繼承實現多型行為（`PlayerControl` 使用鍵盤輸入，`RuleBasedAIControl` 調用規則邏輯，`DQNAIControl` 使用 DQN 模型）。
  - **多型**：`ControlManager` 利用多型動態切換控制策略，透過 `current_strategy` 屬性調用不同實現的 `move` 方法。
  - **封裝**：`ControlManager` 封裝策略實例（`player_control`、`rule_based_ai`、`dqn_ai`）與切換邏輯，提供統一接口（如 `switch_mode`、`move`）。
  - **依賴注入**：`DQNAIControl` 依賴 `DQNAgent`，將模型初始化與狀態生成分離，提升模組化。

#### **`entities/`**
- **`entity_base.py`**
  - **功能**：定義實體基類，提供移動邏輯。
  - **類別**：`EntityBase`
  - **OOP 應用**：
    - **繼承**：作為 `PacMan` 和 `Ghost` 和 `Pellets` 的基類，提供共用屬性（位置、目標）與方法（`move_towards_target`）。
    - **封裝**：封裝移動邏輯，計算逐像素移動（基於 FPS 和 CELL_SIZE），確保平滑動畫。
    - **抽象化**：定義通用接口，允許子類自訂行為。

- **`entity_initializer.py`**
  - **功能**：初始化遊戲實體（Pac-Man、鬼魂、能量球、分數球）。
  - **OOP 應用**：
    - **物件創建**：透過靜態方法生成實體實例，確保隨機位置符合迷宮規則。
    - **模組化**：與 `Game` 類別協作，封裝初始化邏輯，提升可讀性。

- **`ghost.py`**
  - **功能**：實現鬼魂行為，支援多種追逐策略。
  - **類別**：`Ghost`（基類）與子類（`Ghost1`、`Ghost2`、`Ghost3`、`Ghost4`）
  - **OOP 應用**：
    - **繼承**：繼承 `EntityBase`
    - **繼承與多型**：`Ghost` 提供通用行為（BFS 尋路、可食用狀態），子類實現獨特策略（如預測 Pac-Man 位置、側翼包抄）。
    - **封裝**：封裝鬼魂狀態（`edible`、`returning_to_spawn`）與行為（`move`、`update`）。
    - **動態行為**：透過屬性（`edible_timer`、`alpha`）實現狀態切換與視覺效果。

- **`pacman.py`**
  - **功能**：定義 Pac-Man 實體，包含移動與規則 AI 邏輯。
  - **類別**：`PacMan`
  - **OOP 應用**：
    - **繼承**：繼承 `EntityBase`，重用移動邏輯。
    - **封裝**：封裝得分（`score`）、移動邏輯（`set_new_target`）與 AI 行為（`rule_based_ai_move`）。
    - **演算法實現**：使用 A* 演算法實現規則 AI，優先收集分數球並避開危險鬼魂。

- **`pellets.py`**
  - **繼承**：繼承 `EntityBase`
  - **功能**：定義能量球與分數球。
  - **類別**：`PowerPellet`、`ScorePellet`
  - **OOP 應用**：
    - **封裝**：封裝位置與分數屬性，提供簡單接口供 `Game` 檢測碰撞。
    - **可擴展性**：支援未來添加新類型球（如獎勵球）。

### **3. AI 模組 (`ai/`)**

#### **`agent.py`**
- **功能**：實現 DQN 代理，管理記憶緩衝與訓練邏輯。
- **類別**：`DQNAgent`
- **OOP 應用**：
  - **封裝**：封裝 DQN 模型、記憶緩衝與超參數，提供方法（如 `choose_action`、`learn`）進行交互。
  - **依賴注入**：依賴 `DQN` 模型（`dqn.py`）與 `SumTree`（`sumtree.py`），實現模組化訓練。

#### **`dqn.py`**
- **功能**：定義 Dueling DQN 神經網路模型。
- **類別**：`DQN`
- **OOP 應用**：
  - **封裝**：封裝卷積層與全連接層，實現狀態到 Q 值的映射。
  - **模組化**：支援 NoisyLinear 層，提升探索效率。

#### **`environment.py`**
- **功能**：提供強化學習環境，符合 OpenAI Gym 規範。
- **OOP 應用**：
  - **封裝**：封裝遊戲狀態與獎勵邏輯，提供標準接口（`step`、`reset`）。
  - **物件繼承**：繼承 `Game` 類別，加強為 DQNAI 訓練專用環境。

#### **`sumtree.py`, `train.py`, `plot_metrics.py`, `test_cuda.py`**
- **功能**：支援 DQN 訓練與可視化。
- **OOP 應用**：
  - **封裝**：`SumTree` 類別封裝優先經驗回放邏輯，`train.py` 封裝訓練迴圈，`plot_metrics.py` 封裝圖表生成。
  - **模組化**：各模組獨立運行，支援單獨測試與整合。

### **4. 配置與文件**
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
  - ESC：暫停遊戲。

### **訓練 DQN 代理**
```bash
python .\ai\train.py
python .\ai\train.py --resume --episodes=9000 --early_stop_reward=4000 --pretrain_episodes=100
```

- 選項：
  - `--resume`：從先前模型繼續訓練
  - `--episodes`：訓練回合數（預設`9000`）
  - `--early_stop_reward`：早期停止的獎勵閾值
  - `--pretrain_episodes`：預訓練的專家採樣回合數
- - 輸出：
  - TensorBoard 記錄儲存於 `runs/`。
  - 模型與記憶緩衝儲存為 `pacman_dqn_final.pth` 與 `replay_buffer_final.pkl`。

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
- **迷宮生成**：極少數情況下生成不連通區塊。
- **規則 AI 錯誤**：規則 AI 偶爾徘徊或無法有效食用鬼魂。
- **預計改進**：
  - 增強選單功能（排行榜篩選、設定自訂選項）。
  - 添加動畫效果，改善視覺體驗。

## 💡 作者
由 Group 6 開發，感謝 `members/` 中列出的貢獻者。

## 📜 許可證
本專案採用 MIT 許可證，詳見 `LICENSE` 檔案（需自行添加）。