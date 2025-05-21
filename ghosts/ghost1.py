# ghosts/ghost1.py
"""
定義 Ghost1，繼承自 BasicGhost，使用 BFS 直接追逐 Pac-Man。
"""
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan
from config import RED
from typing import List
import random

class Ghost1(BasicGhost):
    def __init__(self, x: int, y: int, name: str = "Ghost1"):
        """
        初始化 Ghost1，設置位置、名稱和顏色。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱，預設為 "Ghost1"。
        """
        super().__init__(x, y, name, color=RED)

    def chase_pacman(self, pacman: PacMan, maze, ghosts: List['Ghost1'] = None):
        """
        使用 BFS 直接追逐 Pac-Man，若無路徑則嘗試附近目標。
        
        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            ghosts (List[Ghost1]): 其他鬼魂列表（未使用）。
        """
        direction = self.bfs_path(self.x, self.y, pacman.x, pacman.y, maze)
        if direction:
            if self.set_new_target(direction[0], direction[1], maze):
                self.last_x, self.last_y = self.x, self.y
                return
        
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
                return
        
        self.move_random(maze)