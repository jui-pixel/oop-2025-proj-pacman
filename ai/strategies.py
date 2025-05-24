# ai/strategies.py
"""
定義 Pac-Man 的控制策略，包括玩家控制、規則基礎 AI 和 DQN AI。
提供切換控制模式的功能，支援鍵盤輸入和自動化 AI 控制。
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import pygame
try:
    from .agent import DQNAgent
    import torch
    import numpy as np
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not found. AI mode will use rule-based AI instead.")

class ControlStrategy(ABC):
    @abstractmethod
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        抽象方法，定義移動邏輯。

        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否繼續移動。
        """
        pass

class PlayerControl(ControlStrategy):
    def __init__(self):
        """初始化玩家控制，設置初始移動方向為靜止。"""
        self.dx, self.dy = 0, 0

    def handle_event(self, event) -> None:
        """
        處理鍵盤輸入，更新移動方向。

        Args:
            event (pygame.event.Event): Pygame 事件。
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.dx, self.dy = 0, -1
            elif event.key == pygame.K_DOWN:
                self.dx, self.dy = 0, 1
            elif event.key == pygame.K_LEFT:
                self.dx, self.dy = -1, 0
            elif event.key == pygame.K_RIGHT:
                self.dx, self.dy = 1, 0

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        根據玩家輸入移動 Pac-Man。

        Args:
            pacman, maze, power_pellets, score_pellets, ghosts: 參見 ControlStrategy.move。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始移動。
        """
        if not moving and (self.dx != 0 or self.dy != 0):
            if pacman.set_new_target(self.dx, self.dy, maze):
                moving = True
        return moving

class RuleBasedAIControl(ControlStrategy):
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """
        使用規則基礎的 AI 移動 Pac-Man，調用 Pac-Man 的內建 AI 邏輯。

        Args:
            pacman, maze, power_pellets, score_pellets, ghosts: 參見 ControlStrategy.move。
            moving (bool): 是否正在移動。

        Returns:
            bool: 是否開始移動。
        """
        if not moving:
            moving = pacman.rule_based_ai_move(maze, power_pellets, score_pellets, ghosts)
        return moving

class DQNAIControl(ControlStrategy):
    def __init__(self, maze_width: int, maze_height: int):
        """
        初始化 DQN AI 控制，載入預訓練模型。

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
        self.agent = DQNAgent((maze_height, maze_width, 6), 4, self.device, 10000, 128, 1e-4, 0.0)
        try:
            self.agent.load("pacman_dqn_final.pth")
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
            bool: 是否開始移動。
        """
        if not moving:
            # 構建當前狀態
            state = np.zeros((maze.h, maze.w, 6), dtype=np.float32)
            state[pacman.x, pacman.y, 0] = 1  # Pac-Man 位置
            for pellet in power_pellets:
                state[pellet.x, pellet.y, 1] = 1  # 能量球位置
            for pellet in score_pellets:
                state[pellet.x, pellet.y, 2] = 1  # 分數球位置
            for ghost in ghosts:
                if ghost.edible and ghost.respawn_timer > 0:
                    state[ghost.x, ghost.y, 3] = 1  # 可吃鬼魂
                else:
                    state[ghost.x, ghost.y, 4] = 1  # 不可吃鬼魂
            # 加入牆壁位置 (通道 5)
            for y in range(self.maze.h):
                for x in range(self.maze.w):
                    if self.maze.get_tile(x, y) in ['#', 'X', 'D']:  # 表示牆壁
                        state[x, y, 5] = 1.0
                
            # 選擇動作
            action = self.agent.get_action(state)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]  # 將動作映射到方向
            if pacman.set_new_target(dx, dy, maze):
                moving = True
        return moving

class ControlManager:
    def __init__(self, maze_width: int, maze_height: int):
        """
        初始化控制管理器，管理不同控制策略並支援模式切換。

        Args:
            maze_width (int): 迷宮寬度。
            maze_height (int): 迷宮高度。
        """
        self.player_control = PlayerControl()
        self.rule_based_ai = RuleBasedAIControl()
        self.dqn_ai = None
        if PYTORCH_AVAILABLE:
            try:
                self.dqn_ai = DQNAIControl(maze_width, maze_height)
            except (FileNotFoundError, ImportError) as e:
                print(f"DQN AI initialization failed: {e}")
                print("Falling back to rule-based AI.")
        self.current_strategy = self.player_control
        self.moving = False

    def switch_mode(self):
        """在玩家控制和 AI 控制之間切換模式。"""
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
            self.switch_mode()
        if self.current_strategy == self.player_control:
            self.player_control.handle_event(event)

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts) -> bool:
        """
        執行當前策略的移動邏輯，並更新移動狀態。

        Args:
            pacman, maze, power_pellets, score_pellets, ghosts: 參見 ControlStrategy.move。

        Returns:
            bool: 是否正在移動。
        """
        self.moving = self.current_strategy.move(pacman, maze, power_pellets, score_pellets, ghosts, self.moving)
        if self.moving and pacman.move_towards_target(maze):
            self.moving = False
        return self.moving

    def get_mode_name(self) -> str:
        """
        返回當前控制模式的顯示名稱。

        Returns:
            str: 模式名稱。
        """
        if self.current_strategy == self.player_control:
            return "Player Mode"
        return "DQN AI Mode" if self.dqn_ai and self.current_strategy == self.dqn_ai else "Rule AI Mode"