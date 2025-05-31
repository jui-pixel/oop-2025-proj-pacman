# game/entities/basic_ghost.py
"""
定義基礎鬼魂類別，提供通用行為（如 BFS 路徑尋找、隨機移動等）。
子類可覆寫 chase_pacman 方法實現特定追逐策略。
"""
from .entity_base import Entity
from ..maze_generator import Map
from typing import Tuple, List, Optional
from collections import deque
import random
from config import *

class Ghost(Entity):
    def __init__(self, x: int, y: int, name: str = "Ghost", color: Tuple[int, int, int] = RED):
        """
        初始化基礎鬼魂，設置位置、名稱和顏色。

        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱。
            color (Tuple[int, int, int]): 鬼魂顏色，預設為紅色。
        """
        super().__init__(x, y, 'G')
        self.name = name
        self.color = color
        self.default_speed = 2.0
        self.speed = 2.0
        self.edible = False
        self.edible_timer = 0
        self.returning_to_spawn = False
        self.return_speed = 5.5
        self.death_count = 0
        self.waiting = False
        self.wait_timer = 0
        self.alpha = 255
        self.last_x = None
        self.last_y = None

    def move(self, pacman, maze, fps: int, ghosts: List['Ghost'] = None):
        """
        根據鬼魂狀態執行移動邏輯。

        Args:
            pacman: Pac-Man 物件。
            maze: 迷宮物件。
            fps (int): 每秒幀數。
            ghosts (List[BasicGhost]): 其他鬼魂列表，用於協同追逐。
        """
        if self.waiting:
            self.wait_timer -= 1
            if self.wait_timer <= 0:
                self.waiting = False
                self.speed = self.default_speed
            return

        if self.edible:
            self.edible_timer -= 1
            if self.edible_timer <= 0:
                self.edible = False
                self.edible_timer = 0

        if self.returning_to_spawn:
            self.return_to_spawn(maze)
        elif self.edible and self.edible_timer > 0:
            self.escape_from_pacman(pacman, maze)
        else:
            self.chase_pacman(pacman, maze, ghosts)

    def bfs_path(self, start_x: int, start_y: int, target_x: int, target_y: int, maze) -> Optional[Tuple[int, int]]:
        """
        使用 BFS 尋找從 (start_x, start_y) 到 (target_x, target_y) 的最短路徑。

        Args:
            start_x (int): 起始 x 座標。
            start_y (int): 起始 y 座標。
            target_x (int): 目標 x 座標。
            target_y (int): 目標 y 座標。
            maze: 迷宮物件。

        Returns:
            Optional[Tuple[int, int]]: 第一步的方向 (dx, dy) 或 None。
        """
        queue = deque([(start_x, start_y, [])])
        visited = {(start_x, start_y)}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            x, y, path = queue.popleft()
            if x == target_x and y == target_y:
                return path[0] if path else None

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (maze.xy_valid(new_x, new_y) and
                    maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S'] and
                    (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    new_path = path + [(dx, dy)]
                    queue.append((new_x, new_y, new_path))

        return None

    def move_to_target(self, target_x: int, target_y: int, maze) -> bool:
        """
        嘗試移動到目標位置，若無路徑則嘗試附近目標。

        Args:
            target_x (int): 目標 x 座標。
            target_y (int): 目標 y 座標。
            maze: 迷宮物件。

        Returns:
            bool: 是否成功設置目標。
        """
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        direction = self.bfs_path(self.x, self.y, target_x, target_y, maze)
        if direction and self.set_new_target(direction[0], direction[1], maze):
            self.last_x, self.last_y = self.x, self.y
            return True

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
                return True
        return False

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        基礎追逐邏輯，使用 BFS 追逐 Pac-Man，若無路徑則嘗試附近目標。

        Args:
            pacman: Pac-Man 物件。
            maze: 迷宮物件。
            ghosts (List[BasicGhost]): 其他鬼魂列表，用於協同追逐。
        """
        pass

    def return_to_spawn(self, maze):
        """
        快速返回最近的重生點 'S'。

        Args:
            maze: 迷宮物件。
        """
        self.speed = self.return_speed
        spawn_points = [(x, y) for y in range(maze.height)
                        for x in range(maze.width) if maze.get_tile(x, y) == 'S']
        if not spawn_points:
            self.move_random(maze)
            return

        closest_spawn = min(spawn_points, key=lambda p: (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2)
        direction = self.bfs_path(self.x, self.y, closest_spawn[0], closest_spawn[1], maze)
        if direction and self.set_new_target(direction[0], direction[1], maze):
            self.last_x, self.last_y = self.x, self.y
            return
        self.move_random(maze)

    def escape_from_pacman(self, pacman, maze):
        """
        在可吃狀態下逃離 Pac-Man。

        Args:
            pacman: Pac-Man 物件。
            maze: 迷宮物件。
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        max_distance = -1

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'E', 'S', 'D']:
                distance = ((new_x - pacman.x) ** 2 + (new_y - pacman.y) ** 2) ** 0.5
                if distance > max_distance:
                    max_distance = distance
                    best_direction = (dx, dy)

        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            return
        self.move_random(maze)

    def move_random(self, maze):
        """
        隨機選擇一個可通行方向移動。

        Args:
            maze: 迷宮物件。
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.set_new_target(dx, dy, maze):
                return

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """
        設置新目標格子，檢查是否可通行並避免反覆移動。

        Args:
            dx (int): x 方向偏移。
            dy (int): y 方向偏移。
            maze: 迷宮物件。

        Returns:
            bool: 是否成功設置目標。
        """
        new_x, new_y = self.x + dx, self.y + dy
        if (self.last_x is not None and new_x == self.last_x and new_y == self.last_y and
            random.random() < 0.8):
            return False
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S']:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

    def set_edible(self, duration: int):
        """
        設置鬼魂為可吃狀態。

        Args:
            duration (int): 可吃持續時間（幀數）。
        """
        if not (self.returning_to_spawn or self.waiting):
            self.edible = True
            self.edible_timer = duration

    def set_returning_to_spawn(self, fps: int):
        """
        設置鬼魂返回重生點。

        Args:
            fps (int): 每秒幀數。
        """
        self.speed = self.return_speed
        self.death_count += 1
        self.returning_to_spawn = True
        self.edible = False
        self.edible_timer = 0
        self.alpha = 255

    def set_waiting(self, fps: int):
        """
        設置鬼魂為等待狀態。

        Args:
            fps (int): 每秒幀數。
        """
        self.returning_to_spawn = False
        self.waiting = True
        wait_time = 10.0 / max(1, self.death_count)
        self.wait_timer = int(wait_time * 900 / fps)

    def reset(self, maze : Map):
        """
        重置鬼魂狀態。
        """
        self.returning_to_spawn = False
        self.edible = False
        self.edible_timer = 0
        self.waiting = False
        self.wait_timer = 0
        self.last_x, self.last_y = None, None
        spawn_points = [(x, y) for y in range(maze.height)
                        for x in range(maze.width) if maze.get_tile(x, y) == 'S']
        if spawn_points:
            self.x, self.y = random.choice(spawn_points)
        

# 子類定義
class Ghost1(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost1"):
        """
        初始化 Ghost1，使用紅色。

        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱，預設為 "Ghost1"。
        """
        super().__init__(x, y, name, color=RED)

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        使用 BFS 直接追逐 Pac-Man，若無路徑則嘗試附近目標。

        Args:
            pacman: Pac-Man 物件。
            maze: 迷宮物件。
            ghosts (List[BasicGhost]): 其他鬼魂列表（未使用）。
        """
        if self.move_to_target(pacman.x, pacman.y, maze):
            return

class Ghost2(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost2"):
        """
        初始化 Ghost2，使用粉紅色。

        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱，預設為 "Ghost2"。
        """
        super().__init__(x, y, name, color=PINK)

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        預測 Pac-Man 移動方向，瞄準其前方 4 格的位置。

        Args:
            pacman: Pac-Man 物件。
            maze: 迷宮物件。
        """
        dx, dy = 0, 0
        if pacman.target_x > pacman.x:
            dx = 1
        elif pacman.target_x < pacman.x:
            dx = -1
        elif pacman.target_y > pacman.y:
            dy = 1
        elif pacman.target_y < pacman.y:
            dy = -1
        target_x = pacman.x + dx * 4
        target_y = pacman.y + dy * 4
        if self.move_to_target(target_x, target_y, maze):
            return

class Ghost3(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost3"):
        """
        初始化 Ghost3，使用青色。

        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱，預設為 "Ghost3"。
        """
        super().__init__(x, y, name, color=CYAN)

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        與 Ghost1 協同，瞄準 Pac-Man 前方 2 格與 Ghost1 的對稱點。

        Args:
            pacman: Pac-Man 物件。
            maze: 迷宮物件。
            ghosts (List[BasicGhost]): 其他鬼魂列表。
        """
        ghost1 = next((g for g in ghosts if g.name == "Ghost1"), None) if ghosts else None
        if not ghost1:
            self.move_random(maze)
            return
        dx, dy = 0, 0
        if pacman.target_x > pacman.x:
            dx = 1
        elif pacman.target_x < pacman.x:
            dx = -1
        elif pacman.target_y > pacman.y:
            dy = 1
        elif pacman.target_y < pacman.y:
            dy = -1
        mid_x = pacman.x + dx * 2
        mid_y = pacman.y + dy * 2
        target_x = ghost1.x + 2 * (mid_x - ghost1.x)
        target_y = ghost1.y + 2 * (mid_y - ghost1.y)
        if self.move_to_target(target_x, target_y, maze):
            return

class Ghost4(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost4"):
        """
        初始化 Ghost4，使用淺藍色。

        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱，預設為 "Ghost4"。
        """
        super().__init__(x, y, name, color=LIGHT_BLUE)

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        當與 Pac-Man 距離小於 8 格時隨機移動，否則追逐 Pac-Man。

        Args:
            pacman: Pac-Man 物件。
            maze: 迷宮物件。
        """
        distance = ((self.x - pacman.x) ** 2 + (self.y - pacman.y) ** 2) ** 0.5
        threshold = 8
        if distance < threshold:
            self.move_random(maze)
            return
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        min_distance = float('inf')
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S']:
                distance = ((new_x - pacman.x) ** 2 + (new_y - pacman.y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    best_direction = (dx, dy)
        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            return