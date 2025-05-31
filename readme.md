# Pac-Man AI 專案

這是一個基於 Pygame 的 Pac-Man 遊戲實現，結合深度 Q 學習（DQN）與規則基礎的 AI 控制策略。專案支援隨機迷宮生成、多樣化的鬼魂行為、玩家控制以及 AI 自動控制，並提供 DQN 代理訓練與訓練結果可視化工具。

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

## 🛠️ 模組詳細說明

### **1. 主程式**
- `main.py`  
  遊戲入口，初始化 Pygame，設置遊戲視窗，處理事件並渲染畫面。透過 `strategies.py` 切換玩家控制、規則基礎 AI 與 DQN AI 模式。

### **2. 遊戲邏輯模組 (`game/`)**  
- `entity_base.py`  
  定義遊戲實體基類，提供基本移動邏輯（逐像素移動至目標格子）與目標設置功能，確保所有實體（如 Pac-Man、鬼魂）具有統一的移動行為。  
- `entity_initializer.py`  
  初始化遊戲實體，包括 Pac-Man、鬼魂、能量球和分數球。Pac-Man 隨機生成於邊緣非中間位置，鬼魂生成於重生點（'S'），能量球和分數球分佈於有效路徑格子。  
- `ghost.py`  
  實現鬼魂行為邏輯，包含基礎鬼魂類與四種子類（Ghost1 至 Ghost4）。支援 BFS 路徑尋找、隨機移動、可食用狀態逃跑、返回重生點等。子類實現獨特追逐策略，如預測 Pac-Man 位置、側翼包抄、協同設陷阱等。  
- `maze_generator.py`  
  生成隨機對稱迷宮，包含牆壁（'X'）、路徑（'.'）、能量球（'E'）、鬼魂重生點（'S'）和門（'D'）。使用牆壁擴展與路徑縮窄演算法，確保連通性與挑戰性，能量球均勻分佈。  
- `menu.py`  
  提供遊戲選單介面，包括主選單、排行榜、設定頁面、暫停選單與遊戲結果顯示。支援滑鼠與鍵盤操作，儲存分數至 `scores.json`，並處理玩家名稱輸入。  
- `pacman.py`  
  定義 Pac-Man 實體，包含移動、得分與規則基礎 AI 邏輯。使用 A* 演算法規劃路徑，優先收集分數球，避開危險鬼魂，追逐可食用鬼魂。包含脫困機制與連續上下交替移動檢測，提升 AI 表現。  
- `pellets.py`  
  定義能量球（PowerPellet，10 分）與分數球（ScorePellet，2 分），提供位置與分數屬性，支援 Pac-Man 吃球邏輯。  
- `renderer.py`  
  負責遊戲畫面渲染，包括迷宮、Pac-Man、鬼魂、能量球、分數球與分數/模式顯示。支援動態鬼魂透明度（可食用或返回重生點時閃爍），確保視覺效果清晰。  
- `strategies.py`  
  實現控制策略，包括玩家鍵盤控制、規則基礎 AI 與 DQN AI。支援動態切換模式，DQN AI 使用 6 通道狀態（Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂、牆壁）進行動作選擇。  
- `game.py`  
  核心遊戲邏輯模組，管理遊戲狀態更新、碰撞檢測與實體互動。支援 Pac-Man 吃球、鬼魂追逐/被吃、遊戲結束條件（全吃或被鬼魂抓住），並儲存分數與遊玩數據。

### **3. AI 模組 (`ai/`)**
- `agent.py`  
  實現 DQN 代理，支援優先經驗回放（Prioritized Experience Replay）。包含增強的 epsilon-greedy 探索策略（線性衰減與預熱階段），用於平衡探索與利用。提供模型與記憶緩衝的儲存/載入功能，與 `dqn.py` 和 `sumtree.py` 協同工作。
- `dqn.py`  
  定義 Dueling DQN 神經網路模型，採用卷積層提取 6 通道遊戲狀態特徵（Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂、牆壁），並透過價值流與優勢流預測動作 Q 值，提升動作選擇效率。
- `environment.py`  
  符合 OpenAI Gym 規範的強化學習環境，專為 DQN 訓練設計。支援可切換渲染（透過 `--visualize` 啟用），提供 6 通道狀態表示，並計算獎勵（吃球、吃鬼、時間懲罰、遊戲結束）。修復了 Pac-Man 被鬼魂吃掉後回合未重啟的錯誤，確保正確的生命數追蹤與環境重置。
- `sumtree.py`  
  實現 SumTree 數據結構，支援優先經驗回放，用於高效儲存與採樣經驗，提升 DQN 訓練效率。
- `train.py`  
  執行 DQN 訓練迴圈，記錄獎勵、損失與 epsilon 值至 TensorBoard（`runs/`）及 `episode_rewards.json`。支援恢復訓練（`--resume`）與可視化（`--visualize`），並定期儲存模型。
- `plot_metrics.py`  
  從 TensorBoard 或 `episode_rewards.json` 提取訓練數據，繪製獎勵與損失曲線，儲存為 `training_metrics.png`，便於離線分析訓練進度。
- `test_cuda.py`  
  檢查 CUDA 可用性並顯示 GPU 資訊，協助驗證 PyTorch 與 CUDA 環境設置。

### **4. 配置與文件**
- `config.py`  
  定義遊戲常量，例如迷宮尺寸、顏色、FPS 與計分規則。
- `docx/classdiagram.md`  
  使用 Mermaid 語法描述類別關係。
- `docx/requirements.txt`  
  列出 Python 依賴。

## 🚀 快速開始

### **環境要求**
- Python 3.13.2（或 3.8 以上）
- Pygame 2.6.1
- PyTorch（支援 CUDA 可選）(若要執行/訓練 DQNAI 必選)
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

### **訓練 DQN 代理**
開始訓練：
```bash
python ai/train.py [--episodes 1000] [--visualize] [--resume] [--render_frequency 10]
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
- **DQNAI渲染錯誤**:visualize=True 時會不明原因當機
- **DQNAI學習錯誤**:目前 DQN AI 只會左右移動
- **迷宮生成錯誤**:極少數情況下會生成不連通的區塊
- **RULEAI巡路錯誤**:少數情況下 RULE AI 會一直徘徊
- **RULEAI鎖敵錯誤**:有時 RULE AI 會吃不到 Ghost

## 預計修改內容
為了提升 Pac-Man AI 專案的遊戲體驗、AI 表現與開發者友好性，我們計劃進行以下改進：
1. **遊戲邏輯優化 (`game/`)**
   - **鬼魂策略多樣化**：加強鬼魂
   - **修正BUG**:
2. **AI 訓練改進 (`ai/`)**
   - **修復DQN、ENV**：或許要全部重新架構
3. **用戶體驗提升**  
   - **排行榜功能擴展**：在 `menu.py` 中增強排行榜功能，支援篩選（按模式或迷宮種子）與歷史記錄導出，提升玩家互動性。  
   - **設定選項**：完善 `menu.py` 中的設定頁面，允許玩家自訂鍵盤映射、音效開關或迷宮參數（如 `MAZE_WIDTH`、`MAZE_HEIGHT`）。  
   - **視覺效果優化**：在 `renderer.py` 中添加動畫效果（如 Pac-Man 吃球動畫、鬼魂狀態轉換漸變），提升視覺吸引力。  
4. **錯誤修復與穩定性** 
   - **資源管理**：優化 `game.py` 與 `renderer.py` 等的 Pygame 初始化與清理邏輯，防止記憶體洩漏或崩潰。  
   - **提升相容性**:或許可兼容CPU訓練/使用。  
5. **新功能探索**  
   - **模組化擴展**：設計插件系統，允許開發者輕鬆添加新鬼魂行為（`ghost.py`）、迷宮類型（`maze_generator.py`）或控制策略（`strategies.py`）。  
## 💡 作者
由 Group 6 開發，感謝 `members/` 中列出的所有貢獻者！

## 📜 許可證
本專案採用 MIT 許可證，詳情見 `LICENSE` 檔案（需自行添加）。