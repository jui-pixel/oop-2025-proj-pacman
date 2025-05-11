pacman_ai/
├── main.py                    # 主程式，啟動遊戲
├── game/
│   ├── __init__.py
│   ├── game_env.py           # Gym 環境實作
│   ├── maze_generator.py     # 隨機迷宮產生器
│   ├── entities.py           # 定義小精靈、鬼魂、能量球等角色行為
│   └── settings.py           # 遊戲設定（尺寸、速度、分數等）
├── ai/
│   ├── __init__.py
│   ├── agent.py              # AI 代理人程式碼
│   ├── dqn.py                # 具體 AI 方法（如 DQN）
│   └── train.py              # 訓練迴圈
└── assets/                   # 音效、圖像等素材