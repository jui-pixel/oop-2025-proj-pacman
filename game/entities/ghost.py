# game/entities/ghost.py
"""
定義基礎鬼魂類別，提供通用行為（如 BFS 路徑尋找、隨機移動等）。
子類可覆寫 chase_pacman 方法實現特定追逐策略。
"""
from .entity_base import Entity
from ..maze_generator import Map
from typing import Tuple, List, Optional
from collections import deque
import random
from config import RED, PINK, CYAN, LIGHT_BLUE, TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN, GHOST_DEFAULT_SPEED, GHOST_RETURN_SPEED, GHOST_WAIT_TIME, GHOST1_SPEED, GHOST2_SPEED, GHOST3_SPEED, GHOST4_SPEED

class Ghost(Entity):
    def __init__(self, x: int, y: int, name: str = "Ghost", color: Tuple[int, int, int] = RED):
        """
        初始化基礎鬼魂，設置位置、名稱和顏色。
        """
        super().__init__(x, y, 'G')
        self.name = name
        self.color = color
        self.default_speed = GHOST_DEFAULT_SPEED
        self.speed = GHOST_DEFAULT_SPEED
        self.edible = False
        self.edible_timer = 0
        self.returning_to_spawn = False
        self.return_speed = GHOST_RETURN_SPEED
        self.death_count = 0
        self.waiting = False
        self.wait_timer = 0
        self.alpha = 255
        self.last_x = None
        self.last_y = None

    def move(self, pacman, maze, fps: int, ghosts: List['Ghost'] = None):
        """
        根據鬼魂狀態執行移動邏輯。
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
                    maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN] and
                    (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    new_path = path + [(dx, dy)]
                    queue.append((new_x, new_y, new_path))

        return None

    def move_to_target(self, target_x: int, target_y: int, maze) -> bool:
        """
        嘗試移動到目標位置，若無路徑則嘗試附近目標。
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
            maze.get_tile(self.x + dx * 3, self.y + dy * 3) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
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
        """
        pass

    def return_to_spawn(self, maze):
        """
        快速返回最近的重生點 'S'。
        """
        self.speed = self.return_speed
        spawn_points = [(x, y) for y in range(maze.height)
                        for x in range(maze.width) if maze.get_tile(x, y) == TILE_GHOST_SPAWN]
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
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        max_distance = -1

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR]:
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
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.set_new_target(dx, dy, maze):
                return

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """
        設置新目標格子，檢查是否可通行並避免反覆移動。
        """
        new_x, new_y = self.x + dx, self.y + dy
        if (self.last_x is not None and new_x == self.last_x and new_y == self.last_y and
            random.random() < 0.8):
            return False
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

    def set_edible(self, duration: int):
        """
        設置鬼魂為可吃狀態。
        """
        if not (self.returning_to_spawn or self.waiting):
            self.edible = True
            self.edible_timer = duration

    def set_returning_to_spawn(self, fps: int):
        """
        設置鬼魂返回重生點。
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
        """
        self.returning_to_spawn = False
        self.waiting = True
        wait_time = GHOST_WAIT_TIME / max(1, self.death_count)
        self.wait_timer = int(wait_time * 900 / fps)

    def reset(self, maze: Map):
        """
        重置鬼魂狀態。
        """
        self.default_speed = GHOST_DEFAULT_SPEED
        self.speed = GHOST_DEFAULT_SPEED
        self.edible = False
        self.edible_timer = 0
        self.returning_to_spawn = False
        self.return_speed = GHOST_RETURN_SPEED
        self.death_count = 0
        self.waiting = False
        self.wait_timer = 0
        self.alpha = 255
        self.last_x = None
        self.last_y = None
        spawn_points = [(x, y) for y in range(maze.height)
                        for x in range(maze.width) if maze.get_tile(x, y) == TILE_GHOST_SPAWN]
        if spawn_points:
            self.x, self.y = random.choice(spawn_points)
        self.target_x = self.x
        self.target_y = self.y

# 子類定義
class Ghost1(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost1"):
        """
        初始化 Ghost1，使用紅色。
        """
        super().__init__(x, y, name, color=RED)
        self.speed = GHOST1_SPEED  # 提高速度以增加挑戰性

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        改進追逐邏輯：根據 Pac-Man 最近兩步的移動方向預測其下一位置，優先使用 BFS 追逐，若無路徑則嘗試附近目標。
        """
        # 預測 Pac-Man 的下一位置
        dx, dy = 0, 0
        if pacman.target_x != pacman.x or pacman.target_y != pacman.y:
            dx = pacman.target_x - pacman.x
            dy = pacman.target_y - pacman.y
        target_x = pacman.x + dx * 2  # 預測 Pac-Man 前進兩格
        target_y = pacman.y + dy * 2
        # 限制目標在迷宮內
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        # 使用 BFS 追逐預測位置
        if self.move_to_target(target_x, target_y, maze):
            return
        # 若無路徑，嘗試直接追逐 Pac-Man
        if self.move_to_target(pacman.x, pacman.y, maze):
            return
        # 最後隨機移動
        self.move_random(maze)

class Ghost2(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost2"):
        """
        初始化 Ghost2，使用粉紅色。
        """
        super().__init__(x, y, name, color=PINK)
        self.speed = GHOST2_SPEED  # 提高速度以增加挑戰性

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        改進追逐邏輯：瞄準 Pac-Man 前方 6 格，預測其軌跡，若無路徑則嘗試側翼包抄。
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
        # 瞄準前方 6 格
        target_x = pacman.x + dx * 6
        target_y = pacman.y + dy * 6
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        if self.move_to_target(target_x, target_y, maze):
            return
        # 側翼包抄：嘗試左右兩側的目標
        flank_targets = [
            (pacman.x + dy * 4, pacman.y - dx * 4),  # 左側
            (pacman.x - dy * 4, pacman.y + dx * 4)   # 右側
        ]
        for tx, ty in flank_targets:
            tx = max(0, min(maze.width - 1, tx))
            ty = max(0, min(maze.height - 1, ty))
            if self.move_to_target(tx, ty, maze):
                return
        self.move_random(maze)

class Ghost3(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost3"):
        """
        初始化 Ghost3，使用青色。
        """
        super().__init__(x, y, name, color=CYAN)
        self.speed = GHOST3_SPEED  # 提高速度以增加挑戰性

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        改進追逐邏輯：與 Ghost1 協同，動態計算對稱點，並在接近 Pac-Man 時設置陷阱。
        """
        ghost1 = next((g for g in ghosts if g.name == "Ghost1"), None) if ghosts else None
        if not ghost1:
            self.move_random(maze)
            return
        # 計算 Pac-Man 移動方向
        dx, dy = 0, 0
        if pacman.target_x > pacman.x:
            dx = 1
        elif pacman.target_x < pacman.x:
            dx = -1
        elif pacman.target_y > pacman.y:
            dy = 1
        elif pacman.target_y < pacman.y:
            dy = -1
        # 動態對稱點
        mid_x = pacman.x + dx * 2
        mid_y = pacman.y + dy * 2
        target_x = ghost1.x + 2 * (mid_x - ghost1.x)
        target_y = ghost1.y + 2 * (mid_y - ghost1.y)
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        # 接近 Pac-Man 時設置陷阱
        distance_to_pacman = ((self.x - pacman.x) ** 2 + (self.y - pacman.y) ** 2) ** 0.5
        if distance_to_pacman < 5:
            # 選擇附近路口（多於一個可移動方向的格子）作為陷阱
            nearby_points = [
                (self.x + dx, self.y + dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if maze.xy_valid(self.x + dx, self.y + dy) and
                maze.get_tile(self.x + dx, self.y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
            ]
            if nearby_points:
                valid_points = [
                    (x, y) for x, y in nearby_points
                    if sum(maze.xy_valid(x + dx, y + dy) and maze.get_tile(x + dx, y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                           for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]) > 1
                ]
                if valid_points:
                    target_x, target_y = random.choice(valid_points)
                    if self.move_to_target(target_x, target_y, maze):
                        return
        if self.move_to_target(target_x, target_y, maze):
            return
        self.move_random(maze)

class Ghost4(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost4"):
        """
        初始化 Ghost4，使用淺藍色。
        """
        super().__init__(x, y, name, color=LIGHT_BLUE)
        self.speed = GHOST4_SPEED  # 提高速度以增加挑戰性

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        改進追逐邏輯：當距離 Pac-Man 小於 8 格時，移動到附近路口控制區域，否則使用 BFS 追逐。
        """
        distance = ((self.x - pacman.x) ** 2 + (self.y - pacman.y) ** 2) ** 0.5
        threshold = 8
        if distance < threshold:
            # 尋找附近路口（多於一個可移動方向的格子）並移動到離 Pac-Man 最近的路口
            nearby_points = [
                (self.x + dx, self.y + dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                if maze.xy_valid(self.x + dx, self.y + dy) and
                maze.get_tile(self.x + dx, self.y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
            ]
            if nearby_points:
                # 選擇路口：有多於一個可移動方向的格子
                junctions = [
                    (x, y) for x, y in nearby_points
                    if sum(maze.xy_valid(x + dx, y + dy) and maze.get_tile(x + dx, y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                           for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]) > 1
                ]
                if junctions:
                    # 選擇離 Pac-Man 最近的路口
                    target = min(junctions, key=lambda p: ((p[0] - pacman.x) ** 2 + (p[1] - pacman.y) ** 2) ** 0.5)
                    if self.move_to_target(target[0], target[1], maze):
                        return
            # 若無路口，隨機移動
            self.move_random(maze)
            return

        # 當距離大於等於 8 格時，使用 BFS 追逐 Pac-Man
        # 預測 Pac-Man 的下一位置
        dx, dy = 0, 0
        if pacman.target_x != pacman.x or pacman.target_y != pacman.y:
            dx = pacman.target_x - pacman.x
            dy = pacman.target_y - pacman.y
        target_x = pacman.x + dx  # 預測 Pac-Man 前進一格
        target_y = pacman.y + dy
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        if self.move_to_target(target_x, target_y, maze):
            return
        # 若無路徑，嘗試直接追逐 Pac-Man 的當前位置
        if self.move_to_target(pacman.x, pacman.y, maze):
            return
        # 最後隨機移動
        self.move_random(maze)