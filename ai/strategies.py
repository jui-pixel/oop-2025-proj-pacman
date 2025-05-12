# ai/strategies.py
from abc import ABC, abstractmethod
from typing import List, Tuple
import pygame
# 嘗試導入 PyTorch 相關模塊
try:
    from .agent import DQNAgent
    import torch
    import numpy as np
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not found. AI mode will use rule-based AI instead.")

class ControlStrategy(ABC):
    """操控策略的抽象基類。"""
    @abstractmethod
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """移動 Pac-Man。

        Args:
            pacman: Pac-Man 對象。
            maze: 迷宮對象。
            power_pellets: 能量球列表。
            score_pellets: 分數球列表。
            ghosts: 鬼魂列表。
            moving: 是否正在移動。

        Returns:
            bool: 是否正在移動。
        """
        pass

class PlayerControl(ControlStrategy):
    def __init__(self):
        self.dx, self.dy = 0, 0

    def handle_event(self, event) -> None:
        """處理鍵盤事件。"""
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
        """根據鍵盤輸入移動 Pac-Man。"""
        if not moving and (self.dx != 0 or self.dy != 0):
            if pacman.set_new_target(self.dx, self.dy, maze):
                moving = True
        return moving

class RuleBasedAIControl(ControlStrategy):
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """使用規則 AI 移動 Pac-Man。"""
        if not moving:
            moving = pacman.rule_based_ai_move(maze, power_pellets, score_pellets, ghosts)
        return moving

class DQNAIControl(ControlStrategy):
    def __init__(self, maze_width: int, maze_height: int):
        """初始化 DQN AI 策略。"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DQNAgent((maze_width, maze_height, 5), 4, self.device)
        try:
            self.agent.load("pacman_dqn_final.pth")
        except FileNotFoundError:
            raise FileNotFoundError("Model file 'pacman_dqn_final.pth' not found.")

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        """使用 DQN AI 移動 Pac-Man。"""
        if not moving:
            state = np.zeros((maze.w, maze.h, 5))
            state[pacman.x, pacman.y, 0] = 1
            for pellet in power_pellets:
                state[pellet.x, pellet.y, 1] = 1
            for pellet in score_pellets:
                state[pellet.x, pellet.y, 2] = 1
            for ghost in ghosts:
                if ghost.edible and ghost.respawn_timer > 0:
                    state[ghost.x, ghost.y, 3] = 1
                else:
                    state[ghost.x, ghost.y, 4] = 1
            action = self.agent.get_action(state)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            if pacman.set_new_target(dx, dy, maze):
                moving = True
        return moving

class ControlManager:
    def __init__(self, maze_width: int, maze_height: int):
        """初始化操控管理器。"""
        self.player_control = PlayerControl()
        self.rule_based_ai = RuleBasedAIControl()
        self.dqn_ai = None
        if PYTORCH_AVAILABLE:
            try:
                self.dqn_ai = DQNAIControl(maze_width, maze_height)
            except FileNotFoundError as e:
                print(e)
                print("Falling back to rule-based AI.")
        self.current_strategy = self.player_control
        self.moving = False

    def switch_mode(self):
        """切換操控模式。"""
        if self.current_strategy == self.player_control:
            self.current_strategy = self.dqn_ai if self.dqn_ai else self.rule_based_ai
            mode = "AI Mode" if self.dqn_ai else "Rule AI Mode"
        else:
            self.current_strategy = self.player_control
            mode = "Player Mode"
        print(f"Switched to {mode}")

    def handle_event(self, event):
        """處理事件。"""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
            self.switch_mode()
        if self.current_strategy == self.player_control:
            self.player_control.handle_event(event)

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts) -> bool:
        """移動 Pac-Man。"""
        self.moving = self.current_strategy.move(pacman, maze, power_pellets, score_pellets, ghosts, self.moving)
        if self.moving and pacman.move_towards_target(maze):
            self.moving = False
        return self.moving

    def get_mode_name(self) -> str:
        """獲取當前模式名稱。"""
        if self.current_strategy == self.player_control:
            return "Player Mode"
        return "AI Mode" if self.dqn_ai and self.current_strategy == self.dqn_ai else "Rule AI Mode"