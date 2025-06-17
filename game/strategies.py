"""
定義 Pac-Man 的控制策略，包括玩家控制、規則基礎 AI 和 DQN AI。
提供動態切換控制模式的功能，支援鍵盤輸入和自動化 AI 控制。

這個模組負責管理 Pac-Man 的移動邏輯，允許玩家通過鍵盤控制，或使用基於規則的 AI 或深度學習的 DQN AI 自動控制。
"""

# 匯入必要的模組
import pygame  # 用於處理鍵盤輸入和遊戲事件
import os  # 用於操作文件路徑
import sys
# 將項目根目錄添加到系統路徑，確保可以匯入其他模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from abc import ABC, abstractmethod  # 用於定義抽象基類
from typing import List  # 用於型別提示
# 從 config 檔案匯入常數，例如迷宮尺寸和圖塊類型
from config import MAZE_WIDTH, MAZE_HEIGHT, CELL_SIZE, FPS, TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN
# 嘗試匯入 PyTorch 相關模組，用於 DQN AI
try:
    from ai.agent import DQNAgent  # DQN 代理模組
    import torch  # PyTorch 主模組
    import numpy as np  # 用於數值計算
    from torch.amp import autocast  # 用於自動混合精度計算
    PYTORCH_AVAILABLE = True  # 表示 PyTorch 可用
except ImportError as e:
    PYTORCH_AVAILABLE = False  # 表示 PyTorch 不可用
    print("PyTorch not found. AI mode will use rule-based AI instead.")

class ControlStrategy(ABC):
    """
    抽象基類，定義控制策略的接口。

    原理：
    - 所有控制策略（玩家、規則 AI、DQN AI）都必須實現 move 方法。
    - 提供統一的接口，允許 ControlManager 動態切換不同控制模式。
    """
    @abstractmethod
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        抽象方法，定義 Pac-Man 的移動邏輯。

        原理：
        - 根據當前遊戲狀態（迷宮、Pac-Man、能量球、分數球、鬼魂）決定移動方向。
        - 返回是否繼續移動，影響遊戲主循環的移動狀態追蹤。

        Args:
            pacman (PacMan): Pac-Man 物件，包含位置和移動方法。
            maze (Map): 迷宮物件，包含地圖結構。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
            moving (bool): 是否正在移動（用於追蹤是否需要新目標）。

        Returns:
            bool: 是否繼續移動（True 表示正在朝目標移動）。
        """
        pass

class PlayerControl(ControlStrategy):
    """
    玩家控制策略，通過鍵盤輸入控制 Pac-Man。

    原理：
    - 監聽鍵盤事件（上下左右鍵），更新移動方向 (dx, dy)。
    - 根據方向設置 Pac-Man 的新目標格子，確保移動有效。
    """
    def __init__(self):
        """
        初始化玩家控制策略。

        原理：
        - 設置初始移動方向為靜止 (dx=0, dy=0)。
        """
        self.dx, self.dy = 0, 0  # 移動方向（x, y 增量）

    def handle_event(self, event):
        """
        處理鍵盤輸入事件，更新移動方向。

        原理：
        - 監聽 KEYDOWN 事件，根據按鍵（上下左右）設置 dx, dy。
        - 僅在玩家控制模式下有效，由 ControlManager 調用。

        Args:
            event (pygame.event.Event): Pygame 事件物件，包含按鍵信息。
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.dx, self.dy = 0, -1  # 向上移動
            elif event.key == pygame.K_DOWN:
                self.dx, self.dy = 0, 1  # 向下移動
            elif event.key == pygame.K_LEFT:
                self.dx, self.dy = -1, 0  # 向左移動
            elif event.key == pygame.K_RIGHT:
                self.dx, self.dy = 1, 0  # 向右移動

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        根據鍵盤輸入移動 Pac-Man。

        原理：
        - 若 Pac-Man 已到達當前目標格子（move_towards_target 返回 True），則根據 dx, dy 設置新目標。
        - 使用 pacman.set_new_target 檢查新目標是否有效（例如非牆壁）。
        - 返回移動狀態，指示是否開始新移動。

        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始新移動（True 表示已設置新目標）。
        """
        if pacman.move_towards_target(FPS):  # 若到達當前目標格子
            if self.dx != 0 or self.dy != 0:  # 若有新輸入方向
                if pacman.set_new_target(self.dx, self.dy, maze):  # 設置新目標
                    pacman.last_direction = (self.dx, self.dy)  # 更新最後方向
                    return True
        return moving

class RuleBasedAIControl(ControlStrategy):
    """
    規則基礎 AI 控制策略，根據預設規則移動 Pac-Man。

    原理：
    - 調用 PacMan 物件的 rule_based_ai_move 方法，實現基於規則的移動邏輯。
    - 規則可能包括優先收集能量球、避開鬼魂等。
    """
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        使用規則基礎 AI 移動 Pac-Man。

        原理：
        - 若 Pac-Man 已到達當前目標格子，調用 rule_based_ai_move 選擇新目標。
        - 返回移動狀態，指示是否開始新移動。

        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始新移動（True 表示已設置新目標）。
        """
        if pacman.move_towards_target(FPS):  # 若到達當前目標格子
            return pacman.rule_based_ai_move(maze, power_pellets, score_pellets, ghosts)  # 執行規則 AI 移動
        return moving

class DQNAIControl(ControlStrategy):
    """
    DQN AI 控制策略，使用深度 Q 學習模型控制 Pac-Man。

    原理：
    - 使用預訓練的 DQN 模型（DQNAgent）根據遊戲狀態選擇最佳動作。
    - 狀態為 6 通道的迷宮表示（Pac-Man、能量球、分數球、可食用鬼魂、危險鬼魂、牆壁）。
    - 動作對應四個方向（上、下、左、右）。
    """
    def __init__(self, maze_width: int, maze_height: int, model_path: str = "pacman_dqn_final.pth"):
        """
        初始化 DQN AI 控制策略。

        原理：
        - 初始化 DQN 代理（DQNAgent），設置狀態維度、動作維度和其他超參數。
        - 檢查 PyTorch 可用性並選擇設備（GPU 或 CPU）。
        - 載入預訓練模型，若模型文件不存在則報錯。

        Args:
            maze_width (int): 迷宮寬度（格子數）。
            maze_height (int): 迷宮高度（格子數）。
            model_path (str): DQN 模型文件路徑（預設為 "pacman_dqn_final.pth"）。

        Raises:
            ImportError: 若 PyTorch 不可用。
            FileNotFoundError: 若模型文件不存在。
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN AI.")
        # 選擇計算設備（GPU 或 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化 DQN 代理
        self.agent = DQNAgent(
            state_dim=(6, maze_height, maze_width),  # 狀態維度：6 通道 x 高度 x 寬度
            action_dim=4,  # 動作維度：4 個方向（上、下、左、右）
            device=self.device,
            buffer_size=100000,  # 經驗回放緩衝區大小
            batch_size=32,  # 訓練批次大小
            lr=5e-4,  # 學習率
            gamma=0.97,  # 折扣因子
            target_update_freq=1,  # 目標網絡更新頻率
            n_step=1,  # n 步回報
            alpha=0.73,  # 優先經驗回放的 alpha 參數
            beta=0.486,  # 優先經驗回放的 beta 參數
            beta_increment=0.003,  # beta 增量
            expert_prob_start=0.0,  # 初始專家策略概率
            expert_prob_end=0.0,  # 最終專家策略概率
            expert_prob_decay_steps=1,  # 專家策略衰減步數
            sigma=10.01,  # NoisyLinear 層的噪聲參數
        )
        try:
            self.agent.load(model_path)  # 載入模型
        except FileNotFoundError:
            print(f"Model file '{model_path}' not found. Please train the model first.")
            raise

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        使用 DQN AI 移動 Pac-Man。

        原理：
        - 若 Pac-Man 到達當前目標格子，生成當前遊戲狀態（6 通道迷宮表示）。
        - 使用 DQN 模型選擇最佳動作（0=上, 1=下, 2=左, 3=右）。
        - 將動作轉換為方向 (dx, dy)，並設置新目標。
        - 狀態生成邏輯包含 Pac-Man、能量球、分數球、可食用鬼魂、危險鬼魂和牆壁的位置。

        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始新移動（True 表示已設置新目標）。
        """
        if pacman.move_towards_target(FPS):  # 若到達當前目標格子
            # 生成狀態，6 通道迷宮表示
            state = np.zeros((6, maze.height, maze.width), dtype=np.float32)
            for y in range(maze.height):
                for x in range(maze.width):
                    if maze.get_tile(x, y) in [TILE_BOUNDARY, TILE_WALL]:
                        state[5, y, x] = 1.0  # 牆壁和邊界
            state[0, pacman.y, pacman.x] = 1.0  # Pac-Man 位置
            for pellet in power_pellets:
                state[1, pellet.y, pellet.x] = 1.0  # 能量球位置
            for pellet in score_pellets:
                state[2, pellet.y, pellet.x] = 1.0  # 分數球位置
            for ghost in ghosts:
                if ghost.returning_to_spawn or ghost.waiting:
                    continue
                elif ghost.edible:
                    state[3, ghost.target_y, ghost.target_x] = 1.0  # 可食用鬼魂位置
                else:
                    state[4, ghost.target_y, ghost.target_x] = 1.0  # 危險鬼魂位置
                    state[4, ghost.y, ghost.x] = 1.0
            # 使用自動混合精度進行推斷
            with autocast(self.device.type):
                action = self.agent.choose_action(state)  # 選擇動作
                # print(action)  # 輸出動作（用於調試）
            # 將動作轉換為方向（0=上, 1=下, 2=左, 3=右）
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            if pacman.set_new_target(dx, dy, maze):  # 設置新目標
                pacman.last_direction = (dx, dy)  # 更新最後方向
                return True
        return moving

class ControlManager:
    """
    控制管理器，負責管理不同的控制策略並支援模式切換。

    原理：
    - 實例化玩家控制、規則 AI 和 DQN AI（若可用）策略。
    - 提供切換控制模式、處理事件和執行移動的統一接口。
    - 若 DQN AI 初始化失敗，自動回退到規則 AI。
    """
    def __init__(self, maze_width: int, maze_height: int, model_path: str = "pacman_dqn_final.pth"):
        """
        初始化控制管理器，管理不同的控制策略。

        原理：
        - 初始化玩家控制和規則 AI 策略。
        - 嘗試初始化 DQN AI，若失敗則設置為 None 並回退到規則 AI。
        - 預設控制策略為玩家控制，追蹤移動狀態（moving）。

        Args:
            maze_width (int): 迷宮寬度（格子數）。
            maze_height (int): 迷宮高度（格子數）。
            model_path (str): DQN 模型文件路徑（預設為 "pacman_dqn_final.pth"）。
        """
        self.player_control = PlayerControl()  # 初始化玩家控制策略
        self.rule_based_ai = RuleBasedAIControl()  # 初始化規則 AI 策略
        self.dqn_ai = None  # DQN AI 策略初始為 None
        if PYTORCH_AVAILABLE:
            try:
                self.dqn_ai = DQNAIControl(maze_width, maze_height, model_path)  # 嘗試初始化 DQN AI
            except (FileNotFoundError, ImportError) as e:
                print(f"DQN AI initialization failed: {e}")
                print("Falling back to rule-based AI.")
        self.current_strategy = self.player_control  # 預設為玩家控制
        self.moving = False  # 移動狀態追蹤

    def switch_mode(self):
        """
        在玩家控制和 AI 控制（DQN 或規則基礎）之間切換。

        原理：
        - 若當前為玩家控制，切換到 DQN AI（若可用）或規則 AI。
        - 若當前為 AI 控制，切換回玩家控制。
        - 打印切換後的模式名稱。

        Returns:
            None
        """
        if self.current_strategy == self.player_control:
            self.current_strategy = self.dqn_ai if self.dqn_ai else self.rule_based_ai  # 切換到 AI
            mode = "DQN AI Mode" if self.dqn_ai else "Rule AI Mode"
        else:
            self.current_strategy = self.player_control  # 切換到玩家
            mode = "Player Mode"
        print(f"Switched to {mode}")

    def handle_event(self, event):
        """
        處理 Pygame 事件，僅在玩家控制模式下有效。

        原理：
        - 將事件傳遞給玩家控制策略的 handle_event 方法。
        - 僅在 current_strategy 為 player_control 時執行。

        Args:
            event (pygame.event.Event): Pygame 事件物件。
        """
        if self.current_strategy == self.player_control:
            self.player_control.handle_event(event)

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts) -> None:
        """
        執行 Pac-Man 移動，使用當前控制策略。

        原理：
        - 調用當前控制策略的 move 方法，更新 Pac-Man 的移動狀態。
        - 將移動狀態（self.moving）傳遞給策略，確保連續性。

        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
        """
        self.moving = self.current_strategy.move(pacman, maze, power_pellets, score_pellets, ghosts, self.moving)

    def get_mode_name(self) -> str:
        """
        獲取當前控制模式的名称。

        原理：
        - 根據 current_strategy 返回對應的模式名稱。
        - 區分玩家模式、DQN AI 模式和規則 AI 模式。

        Returns:
            str: 模式名稱（"Player Mode", "DQN AI Mode" 或 "Rule AI Mode"）。
        """
        if self.current_strategy == self.player_control:
            return "Player Mode"
        return "DQN AI Mode" if self.dqn_ai and self.current_strategy == self.dqn_ai else "Rule AI Mode"