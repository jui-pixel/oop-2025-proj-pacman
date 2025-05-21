# ai/strategies.py
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
        pass

class PlayerControl(ControlStrategy):
    def __init__(self):
        self.dx, self.dy = 0, 0

    def handle_event(self, event) -> None:
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
        if not moving and (self.dx != 0 or self.dy != 0):
            if pacman.set_new_target(self.dx, self.dy, maze):
                moving = True
        return moving

class RuleBasedAIControl(ControlStrategy):
    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        if not moving:
            moving = pacman.rule_based_ai_move(maze, power_pellets, score_pellets, ghosts)
        return moving

class DQNAIControl(ControlStrategy):
    def __init__(self, maze_width: int, maze_height: int):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN AI.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DQNAgent((maze_height, maze_width, 5), 4, self.device)
        try:
            self.agent.load("pacman_dqn_final.pth")
        except FileNotFoundError:
            print("Model file 'pacman_dqn_final.pth' not found. Please train the model first.")
            raise

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts, moving: bool) -> bool:
        if not moving:
            state = np.zeros((maze.h, maze.w, 5), dtype=np.float32)
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
        if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
            self.switch_mode()
        if self.current_strategy == self.player_control:
            self.player_control.handle_event(event)

    def move(self, pacman, maze, power_pellets, score_pellets, ghosts) -> bool:
        self.moving = self.current_strategy.move(pacman, maze, power_pellets, score_pellets, ghosts, self.moving)
        if self.moving and pacman.move_towards_target(maze):
            self.moving = False
        return self.moving

    def get_mode_name(self) -> str:
        if self.current_strategy == self.player_control:
            return "Player Mode"
        return "DQN AI Mode" if self.dqn_ai and self.current_strategy == self.dqn_ai else "Rule AI Mode"