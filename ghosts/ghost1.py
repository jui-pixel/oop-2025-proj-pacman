# ghosts/ghost1.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan
from config import RED
from typing import List
import random

class Ghost1(BasicGhost):
    def __init__(self, x: int, y: int, name: str = "Ghost1"):
        super().__init__(x, y, name, color=RED)

    def chase_pacman(self, pacman: PacMan, maze, ghosts: List['Ghost1'] = None):
        """使用 BFS 直接追逐 Pac-Man。"""
        direction = self.bfs_path(self.x, self.y, pacman.x, pacman.y, maze)
        if direction:
            print(f"{self.name} at ({self.x}, {self.y}) moving {direction} to ({pacman.x}, {pacman.y})")
            if self.set_new_target(direction[0], direction[1], maze):
                self.last_x, self.last_y = self.x, self.y
                return

        # 使用備用策略
        print(f"{self.name} no path to Pac-Man, trying nearby target")
        nearby_targets = [
            (self.x + dx * 3, self.y + dy * 3)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if maze.xy_valid(self.x + dx * 3, self.y + dy * 3) and
            maze.get_tile(self.x + dx * 3, self.y + dy * 3) in ['.', 'D', 'E', 'S']
        ]
        if nearby_targets:
            target_x, target_y = random.choice(nearby_targets)
            direction = self.bfs_path(self.x, self.y, target_x, target_y, maze)
            if direction and self.set_new_target(direction[0], direction[1], maze):
                self.last_x, self.last_y = self.x, self.y
                print(f"{self.name} moving to nearby ({target_x}, {target_y})")
                return

        self.move_random(maze)