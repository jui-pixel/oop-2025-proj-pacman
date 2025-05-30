# ghosts/basic_ghost.py
"""
定義基礎鬼魂類別，提供通用行為（如 BFS 路徑尋找、隨機移動等）。
子類可覆寫 chase_pacman 方法實現特定追逐策略。
"""
from game.entities import Ghost, PacMan
from config import RED
from collections import deque
from typing import Tuple, List, Optional
import random

class BasicGhost(Ghost):
    def __init__(self, x: int, y: int, name: str = "BasicGhost", color: Tuple[int, int, int] = RED):
        """
        初始化基礎鬼魂，設置位置、名稱和顏色。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱。
            color (Tuple[int, int, int]): 鬼魂顏色，預設為紅色。
        """
        super().__init__(x, y, name, color=color)
        self.last_x = None
        self.last_y = None
        self.bfs_cache = {}  # 快取 BFS 路徑，提升性能

    def bfs_path(self, start_x: int, start_y: int, target_x: int, target_y: int, maze) -> Optional[Tuple[int, int]]:
        """
        使用 BFS 尋找從 (start_x, start_y) 到 (target_x, target_y) 的最短路徑。
        
        Args:
            start_x (int): 起始 x 座標。
            start_y (int): 起始 y 座標。
            target_x (int): 目標 x 座標。
            target_y (int): 目標 y 座標。
            maze (Map): 迷宮物件。
        
        Returns:
            Optional[Tuple[int, int]]: 第一步的方向 (dx, dy) 或 None。
        """
        cache_key = (start_x, start_y, target_x, target_y)
        if cache_key in self.bfs_cache:
            return self.bfs_cache[cache_key]
        
        queue = deque([(start_x, start_y, [])])
        visited = {(start_x, start_y)}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            x, y, path = queue.popleft()
            if x == target_x and y == target_y:
                direction = path[0] if path else None
                self.bfs_cache[cache_key] = direction
                return direction
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (maze.xy_valid(new_x, new_y) and 
                    maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S'] and 
                    (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    new_path = path + [(dx, dy)]
                    queue.append((new_x, new_y, new_path))
        
        self.bfs_cache[cache_key] = None
        return None

    def chase_pacman(self, pacman: PacMan, maze, ghosts: List['Ghost'] = None):
        """
        基礎追逐邏輯，使用 BFS 追逐 Pac-Man，若無路徑則嘗試附近目標。
        
        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            ghosts (List[Ghost]): 其他鬼魂列表（可選）。
        """
        direction = self.bfs_path(self.x, self.y, pacman.x, pacman.y, maze)
        if direction:
            print(f"{self.name} at ({self.x}, {self.y}) moving {direction} to ({pacman.x}, {pacman.y})")
            if self.set_new_target(direction[0], direction[1], maze):
                self.last_x, self.last_y = self.x, self.y
                return
        
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
        """
        設置新目標格子，檢查是否可通行並避免反覆移動。
        
        Args:
            dx (int): x 方向偏移。
            dy (int): y 方向偏移。
            maze (Map): 迷宮物件。
        
        Returns:
            bool: 是否成功設置目標。
        """
        new_x, new_y = self.x + dx, self.y + dy
        if (self.last_x is not None and new_x == self.last_x and new_y == self.last_y and 
            random.random() < 0.8):
            return False
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S']:
            self.last_x, self.last_y = self.x, self.y
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

    def return_to_spawn(self, maze):
        """
        快速返回最近的重生點 'S'。
        
        Args:
            maze (Map): 迷宮物件。
        """
        spawn_points = [(x, y) for x, y in [(x, y) for y in range(maze.h) 
                                           for x in range(maze.w) if maze.get_tile(x, y) == 'S']]
        closest_spawn = min(spawn_points, key=lambda p: (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2)
        direction = self.bfs_path(self.x, self.y, closest_spawn[0], closest_spawn[1], maze)
        if direction:
            if self.set_new_target(direction[0], direction[1], maze):
                self.last_x, self.last_y = self.x, self.y
                return