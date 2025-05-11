
## 說明

- **main.py**: 啟動遊戲的主程式。
- **game/**: 遊戲邏輯與環境相關的模組。
  - `game_env.py`: 使用 Gym 實作的遊戲環境。
  - `maze_generator.py`: 隨機迷宮產生器。
  - `entities.py`: 定義遊戲中的角色行為（如小精靈、鬼魂、能量球等）。
  - `settings.py`: 遊戲設定（如尺寸、速度、分數等）。
- **ai/**: 與人工智慧相關的模組。
  - `agent.py`: AI 代理人的程式碼。
  - `dqn.py`: 深度強化學習方法（如 DQN）。
  - `train.py`: 訓練 AI 的迴圈。
- **assets/**: 遊戲所需的音效、圖像等素材。

## 使用方式

1. 確保已安裝必要的依賴項目（如 Python 和相關套件）。
2. 執行 `main.py` 啟動遊戲：
   ```bash
   python main.py