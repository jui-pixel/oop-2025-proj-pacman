"""
定義 Pac-Man 的控制策略，包括玩家控制、規則基礎 AI 和 DQN AI。
提供動態切換控制模式的功能，支援鍵盤輸入和自動化 AI 控制。
"""
import pygame
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from abc import ABC, abstractmethod
from typing import List
from config import MAZE_WIDTH, MAZE_HEIGHT, CELL_SIZE, FPS
import pygame
try:
    from ai.agent import DQNAgent
    import torch
    import numpy as np
    PYTORCH_AVAILABLE = True  # 標記 PyTorch 可用
except ImportError as e:
    PYTORCH_AVAILABLE = False  # 無 PyTorch 時回退到規則 AI
    print("PyTorch not found. AI mode will use rule-based AI instead.")

from config import TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN

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
        self.dx, self.dy = 0, 0

    def handle_event(self, event):
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
        if not moving and pacman.move_towards_target(FPS):
            if self.dx != 0 or self.dy != 0:
                if pacman.set_new_target(self.dx, self.dy, maze):
                    moving = True
        return moving

class RuleBasedAIControl(ControlStrategy):
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        if not moving and pacman.move_towards_target(FPS):
            moving = pacman.rule_based_ai_move(maze, power_pellets, score_pellets, ghosts)
        return moving

class DQNAIControl(ControlStrategy):
    def __init__(self, maze_width: int, maze_height: int):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN AI.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DQNAgent((6, maze_height, maze_width), 4, self.device, 10000, 128, 1e-4, 0.01)
        try:
            self.agent.load("pacman_dqn.pth")
            self.agent.epsilon = 0.0
        except FileNotFoundError:
            print("Model file 'pacman_dqn_final.pth' not found. Please train the model first.")
            raise

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        if not moving and pacman.move_towards_target(FPS):
            state = np.zeros((6, maze.height, maze.width), dtype=np.float32)
            for y in range(maze.height):
                for x in range(maze.width):
                    if maze.get_tile(x, y) in ['#', 'X']:
                        state[5, y, x] = 1.0
                    state[0, pacman.y, pacman.x] = 1.0
                    for pellet in power_pellets:
                        state[1, pellet.y, pellet.x] = 1.0
                    for pellet in score_pellets:
                        state[2, pellet.y, pellet.x] = 1.0
                    for ghost in ghosts:
                        if ghost.edible and ghost.edible_timer > 0 and not ghost.returning_to_spawn:
                            state[3, ghost.y, ghost.x] = 1.0
                        else:
                            state[4, ghost.y, ghost.x] = 1.0

            action = self.agent.choose_action(state)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            if pacman.set_new_target(dx, dy, maze):
                moving = True
        return moving

class ControlManager:
    def __init__(self, maze_width: int, maze_height: int):
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
        if self.current_strategy == self.player_control:
            self.current_strategy = self.dqn_ai if self.dqn_ai else self.rule_based_ai
            mode = "DQN AI Mode" if self.dqn_ai else "Rule AI Mode"
        else:
            self.current_strategy = self.player_control
            mode = "Player Mode"
        print(f"Switched to {mode}")

    def handle_event(self, event):
        if self.current_strategy == self.player_control:
            self.player_control.handle_event(event)

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts) -> None:
        if pacman.move_towards_target(FPS):
            self.current_strategy.move(pacman, maze, power_pellets, score_pellets, ghosts, self.moving)
        
        

    def get_mode_name(self) -> str:
        if self.current_strategy == self.player_control:
            return "Player Mode"
        return "DQN AI Mode" if self.dqn_ai and self.current_strategy == self.dqn_ai else "Rule AI Mode"