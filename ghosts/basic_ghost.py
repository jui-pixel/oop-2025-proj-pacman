# ghosts/basic_ghost.py
from game.entities import Ghost, PacMan
from config import RED
from collections import deque
from typing import Tuple, List, Optional
import random

class BasicGhost(Ghost):
    def __init__(self, x: int, y: int, name: str = "BasicGhost", color: Tuple[int, int, int] = RED):
        super().__init__(x, y, name, color=color)
        self.last_x = None
        self.last_y = None
        self.bfs_cache = {}  # 路徑快取

    def bfs_path(self, start_x: int, start_y: int, target_x: int, target_y: int, maze) -> Optional[Tuple[int, int]]:
        """使用 BFS 尋找最短路徑，返回第一步方向。"""
        cache_key = (start_x, start_y, target_x, target_y)
        if cache_key in self.bfs_cache:
            return self.bfs_cache[cache_key]

        queue = deque([(start_x, start_y, [])])
        visited = {(start_x, start_y)}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 下、上、右、左

        while queue:
            x, y, path = queue.popleft()
            if x == target_x and y == target_y:
                direction = path[0] if path else None
                self.bfs_cache[cache_key] = direction
                return direction

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (maze.xy_valid(new_x, new_y) and
                    maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S'] and  # 移除 's'，與迷宮一致
                    (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    new_path = path + [(dx, dy)]
                    queue.append((new_x, new_y, new_path))

        self.bfs_cache[cache_key] = None
        return None

    def chase_pacman(self, pacman: PacMan, maze, ghosts: List['Ghost'] = None):
        """基礎追逐邏輯，子類應覆寫。"""
        direction = self.bfs_path(self.x, self.y, pacman.x, pacman.y, maze)
        if direction:
            print(f"{self.name} at ({self.x}, {self.y}) moving {direction} to ({pacman.x}, {pacman.y})")
            if self.set_new_target(direction[0], direction[1], maze):
                self.last_x, self.last_y = self.x, self.y
                return

        # 備用策略：隨機選擇附近可通行格子
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

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """檢查並設置新目標，減少反覆移動。"""
        new_x, new_y = self.x + dx, self.y + dy
        if (hasattr(self, 'last_x') and self.last_x is not None and
            new_x == self.last_x and new_y == self.last_y and
            random.random() < 0.8):  # 降低限制，允許更多靈活性
            return False
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S']:
            self.last_x, self.last_y = self.x, self.y
            self.target_x, self.target_y = new_x, new_y
            return True
        return False