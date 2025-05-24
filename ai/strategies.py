# ai/strategies.py
"""
定義 Pac-Man 的控制策略，包括玩家控制、規則基礎 AI 和 DQN AI。
提供動態切換控制模式的功能，支援鍵盤輸入和自動化 AI 控制。
"""

from abc import ABC, abstractmethod
from typing import List
import pygame
try:
    from .agent import DQNAgent
    import torch
    import numpy as np
    PYTORCH_AVAILABLE = True  # 標記 PyTorch 可用
except ImportError:
    PYTORCH_AVAILABLE = False  # 無 PyTorch 時回退到規則 AI
    print("PyTorch not found. AI mode will use rule-based AI instead.")

class ControlStrategy(ABC):
    @abstractmethod
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        抽象方法，定義 Pac-Man 的移動邏輯。

        Args:
            pacman (PacMan): Pac-Man 物件，包含位置和移動方法。
            maze (Map): 迷宮物件，包含地圖結構。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否繼續移動（True 表示正在朝目標移動）。
        """
        pass

class PlayerControl(ControlStrategy):
    def __init__(self):
        """
        初始化玩家控制策略，設置初始移動方向為靜止。
        """
        self.dx, self.dy = 0, 0  # 初始方向：無移動

    def handle_event(self, event) -> None:
        """
        處理鍵盤輸入事件，更新移動方向。

        Args:
            event (pygame.event.Event): Pygame 事件，僅處理 KEYDOWN 事件。
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.dx, self.dy = 0, -1  # 上
            elif event.key == pygame.K_DOWN:
                self.dx, self.dy = 0, 1   # 下
            elif event.key == pygame.K_LEFT:
                self.dx, self.dy = -1, 0  # 左
            elif event.key == pygame.K_RIGHT:
                self.dx, self.dy = 1, 0   # 右

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        根據玩家鍵盤輸入移動 Pac-Man。

        Args:
            pacman, maze, power_pellets, score_pellets, ghosts: 參見 ControlStrategy.move。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始或繼續移動。
        """
        if not moving and (self.dx != 0 or self.dy != 0):
            # 當靜止時，嘗試設置新目標方向
            if pacman.set_new_target(self.dx, self.dy, maze):
                moving = True  # 成功設置目標，開始移動
        return moving

class RuleBasedAIControl(ControlStrategy):
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        使用規則基礎的 AI 移動 Pac-Man，調用 Pac-Man 的內建 AI 邏輯。

        Args:
            pacman, maze, power_pellets, score_pellets, ghosts: 參見 ControlStrategy.move。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始或繼續移動。
        """
        if not moving:
            # 當靜止時，調用 Pac-Man 的規則 AI 選擇新目標
            moving = pacman.rule_based_ai_move(maze, power_pellets, score_pellets, ghosts)
        return moving

class DQNAIControl(ControlStrategy):
    def __init__(self, maze_width: int, maze_height: int):
        """
        初始化 DQN AI 控制策略，載入預訓練模型。

        Args:
            maze_width (int): 迷宮寬度。
            maze_height (int): 迷宮高度。

        Raises:
            ImportError: 如果缺少 PyTorch。
            FileNotFoundError: 如果模型檔案不存在。
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN AI.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化 DQN 代理，狀態維度為 (高度, 寬度, 6)，動作數為 4
        self.agent = DQNAgent((maze_height, maze_width, 6), 4, self.device, 10000, 128, 1e-4, 0.01)
        try:
            self.agent.load("pacman_dqn_final.pth")  # 載入預訓練模型
        except FileNotFoundError:
            print("Model file 'pacman_dqn_final.pth' not found. Please train the model first.")
            raise

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        使用 DQN 模型選擇動作並移動 Pac-Man。

        Args:
            pacman, maze, power_pellets, score_pellets, ghosts: 參見 ControlStrategy.move。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始或繼續移動。
        """
        if not moving:
            # 構建當前狀態：6 通道表示 Pac-Man、能量球、分數球等
            state = np.zeros((maze.h, maze.w, 6), dtype=np.float32)
            state[pacman.x, pacman.y, 0] = 1  # Pac-Man 位置
            for pellet in power_pellets:
                state[pellet.x, pellet.y, 1] = 1  # 能量球
            for pellet in score_pellets:
                state[pellet.x, pellet.y, 2] = 1  # 分數球
            for ghost in ghosts:
                if ghost.edible and ghost.respawn_timer > 0:
                    state[ghost.x, ghost.y, 3] = 1  # 可吃鬼魂
                else:
                    state[ghost.x, ghost.y, 4] = 1  # 不可吃鬼魂
            # 標記牆壁
            for y in range(maze.h):
                for x in range(maze.w):
                    if maze.get_tile(x, y) in ['#', 'X', 'D']:
                        state[x, y, 5] = 1.0

            # 使用 DQN 選擇動作
            action = self.agent.get_action(state)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]  # 映射動作到方向
            if pacman.set_new_target(dx, dy, maze):
                moving = True  # 成功設置目標，開始移動
        return moving

class ControlManager:
    def __init__(self, maze_width: int, maze_height: int):
        """
        初始化控制管理器，管理玩家控制、規則 AI 和 DQN AI，並支援模式切換。

        Args:
            maze_width (int): 迷宮寬度。
            maze_height (int): 迷宮高度。
        """
        self.player_control = PlayerControl()  # 初始化玩家控制
        self.rule_based_ai = RuleBasedAIControl()  # 初始化規則 AI
        self.dqn_ai = None  # DQN AI，初始為 None
        if PYTORCH_AVAILABLE:
            try:
                self.dqn_ai = DQNAIControl(maze_width, maze_height)  # 嘗試初始化 DQN AI
            except (FileNotFoundError, ImportError) as e:
                print(f"DQN AI initialization failed: {e}")
                print("Falling back to rule-based AI.")
        self.current_strategy = self.player_control  # 預設為玩家控制
        self.moving = False  # 初始移動狀態

    def switch_mode(self):
        """
        在玩家控制和 AI 控制（DQN 或規則 AI）之間切換。
        """
        if self.current_strategy == self.player_control:
            self.current_strategy = self.dqn_ai if self.dqn_ai else self.rule_based_ai
            mode = "DQN AI Mode" if self.dqn_ai else "Rule AI Mode"
        else:
            self.current_strategy = self.player_control
            mode = "Player Mode"
        print(f"Switched to {mode}")

    def handle_event(self, event):
        """
        處理 Pygame 事件，支援模式切換和玩家輸入。

        Args:
            event (pygame.event.Event): Pygame 事件。
        """
        if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
            self.switch_mode()  # 按 'a' 鍵切換模式
        if self.current_strategy == self.player_control:
            self.player_control.handle_event(event)  # 處理玩家輸入

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts) -> bool:
        """
        執行當前策略的移動邏輯，並更新移動狀態。

        Args:
            pacman, maze, power_pellets, score_pellets, ghosts: 參見 ControlStrategy.move。

        Returns:
            bool: 是否正在移動。
        """
        # 執行當前策略的移動
        self.moving = self.current_strategy.move(pacman, maze, power_pellets, score_pellets, ghosts, self.moving)
        if self.moving and pacman.move_towards_target(maze):
            self.moving = False  # 完成移動後重置狀態
        return self.moving

    def get_mode_name(self) -> str:
        """
        返回當前控制模式的顯示名稱。

        Returns:
            str: 模式名稱（"Player Mode", "DQN AI Mode" 或 "Rule AI Mode"）。
        """
        if self.current_strategy == self.player_control:
            return "Player Mode"
        return "DQN AI Mode" if self.dqn_ai and self.current_strategy == self.dqn_ai else "Rule AI Mode"