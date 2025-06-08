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
        self.speed = self.default_speed
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
        self.memory_x = x  # 記憶 Pac-Man 最近位置
        self.memory_y = y

    def move(self, pacman, maze, fps: int, ghosts: List['Ghost'] = None):
        """
        根據鬼魂狀態執行移動邏輯。
        """
        if self.waiting:
            self.wait_timer -= 1
            if self.wait_timer <= 0:
                self.waiting = False
                self.speed = self.default_speed * (1.1 ** self.death_count)
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
        使用 BFS 尋找從 (start_x, start_y) 到 (target_x, target_y) 的最短路徑，優化路徑選擇。
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

        # 若無直接路徑，嘗試最近的可通行點
        nearby_targets = [
            (new_x, new_y) for dx, dy in directions
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
        ]
        if nearby_targets:
            target = min(nearby_targets, key=lambda p: (p[0] - target_x) ** 2 + (p[1] - target_y) ** 2)
            return self.bfs_path(start_x, start_y, target[0], target[1], maze)
        return None

    def move_to_target(self, target_x: int, target_y: int, maze) -> bool:
        """
        嘗試移動到目標位置，若無路徑則嘗試附近目標並優化路徑。
        """
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        direction = self.bfs_path(self.x, self.y, target_x, target_y, maze)
        if direction and self.set_new_target(direction[0], direction[1], maze):
            self.last_x, self.last_y = self.x, self.y
            return True
        return False

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        基礎追逐邏輯，子類覆寫實現具體策略。
        """
        pass

    def return_to_spawn(self, maze):
        """
        快速返回最近的重生點 'S'，優先選擇最近路徑。
        """
        self.speed = self.return_speed
        spawn_points = [(x, y) for y in range(maze.height)
                        for x in range(maze.width) if maze.get_tile(x, y) == TILE_GHOST_SPAWN]
        if not spawn_points:
            self.move_random(maze)
            return

        closest_spawn = min(spawn_points, key=lambda p: (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2)
        if self.move_to_target(closest_spawn[0], closest_spawn[1], maze):
            if self.x == closest_spawn[0] and self.y == closest_spawn[1]:
                self.returning_to_spawn = False
                self.set_waiting(60)  # 假設 fps = 60
            return
        self.move_random(maze)

    def escape_from_pacman(self, pacman, maze):
        """
        在可吃狀態下逃離 Pac-Man，優先選擇最大距離方向。
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
        隨機選擇一個可通行方向移動，減少反覆移動。
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
            random.random() < 0.9):  # 提高避免反覆移動的機率
            return False
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

    def set_edible(self, duration: int):
        """
        設置鬼魂為可吃狀態，減少逃跑效率。
        """
        if not (self.returning_to_spawn or self.waiting):
            self.edible = True
            self.edible_timer = duration

    def set_returning_to_spawn(self, fps: int):
        """
        設置鬼魂返回重生點，增加速度。
        """
        self.speed = self.return_speed
        self.death_count += 1
        self.returning_to_spawn = True
        self.edible = False
        self.edible_timer = 0
        self.alpha = 255

    def set_waiting(self, fps: int):
        """
        設置鬼魂為等待狀態，隨死亡次數減少等待時間。
        """
        self.returning_to_spawn = False
        self.waiting = True
        wait_time = GHOST_WAIT_TIME / max(1, self.death_count * 1.5)
        self.wait_timer = int(wait_time * 900 / fps)

    def reset(self, maze: Map):
        """
        重置鬼魂狀態。
        """
        self.default_speed = GHOST_DEFAULT_SPEED
        self.speed = self.default_speed
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
        self.memory_x = self.x
        self.memory_y = self.y
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
        初始化 Ghost1，使用紅色，作為領頭追逐者。
        """
        super().__init__(x, y, name, color=RED)
        self.speed = GHOST1_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        領頭追逐：預測 Pac-Man 未來 3 步位置，優先使用 BFS 追逐，若無路徑則協調其他鬼魂。
        """
        # 預測 Pac-Man 移動趨勢
        dx, dy = 0, 0
        if pacman.target_x != pacman.x or pacman.target_y != pacman.y:
            dx = pacman.target_x - pacman.x
            dy = pacman.target_y - pacman.y
        target_x = pacman.x + dx * 3  # 預測 3 步
        target_y = pacman.y + dy * 3
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        if self.move_to_target(target_x, target_y, maze):
            self.memory_x, self.memory_y = pacman.x, pacman.y
            return

        # 若無路徑，追蹤記憶位置
        if self.move_to_target(self.memory_x, self.memory_y, maze):
            return

        # 協調其他鬼魂，通知最近的鬼魂調整位置
        if ghosts:
            closest_ghost = min(ghosts, key=lambda g: g != self and ((g.x - pacman.x) ** 2 + (g.y - pacman.y) ** 2) ** 0.5, default=None)
            if closest_ghost:
                flank_x = pacman.x + (pacman.x - self.x)
                flank_y = pacman.y + (pacman.y - self.y)
                closest_ghost.move_to_target(flank_x, flank_y, maze)
        self.move_random(maze)

class Ghost2(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost2"):
        """
        初始化 Ghost2，使用粉紅色，作為側翼包抄者。
        """
        super().__init__(x, y, name, color=PINK)
        self.speed = GHOST2_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        側翼包抄：根據 Ghost1 位置，動態計算包抄點，優先阻塞 Pac-Man 逃跑路線。
        """
        ghost1 = next((g for g in ghosts if g.name == "Ghost1"), None) if ghosts else None
        if not ghost1:
            if self.move_to_target(pacman.x, pacman.y, maze):
                return
            self.move_random(maze)
            return

        # 計算與 Ghost1 的對稱點作為包抄位置
        dx = pacman.x - ghost1.x
        dy = pacman.y - ghost1.y
        target_x = pacman.x + dx * 2
        target_y = pacman.y + dy * 2
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        if self.move_to_target(target_x, target_y, maze):
            return

        # 阻塞關鍵路口
        nearby_points = [
            (pacman.x + dx, pacman.y + dy) for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if maze.xy_valid(pacman.x + dx, pacman.y + dy) and
            maze.get_tile(pacman.x + dx, pacman.y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
        ]
        if nearby_points:
            target = min(nearby_points, key=lambda p: ((p[0] - pacman.x) ** 2 + (p[1] - pacman.y) ** 2) ** 0.5)
            if self.move_to_target(target[0], target[1], maze):
                return
        self.move_random(maze)

class Ghost3(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost3"):
        """
        初始化 Ghost3，使用青色，作為圍堵者。
        """
        super().__init__(x, y, name, color=CYAN)
        self.speed = GHOST3_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        圍堵策略：與 Ghost1 和 Ghost2 協作，計算 Pac-Man 可能的逃跑路徑並設置陷阱。
        """
        ghost1 = next((g for g in ghosts if g.name == "Ghost1"), None) if ghosts else None
        ghost2 = next((g for g in ghosts if g.name == "Ghost2"), None) if ghosts else None
        if not ghost1 or not ghost2:
            if self.move_to_target(pacman.x, pacman.y, maze):
                return
            self.move_random(maze)
            return

        # 預測 Pac-Man 逃跑方向
        dx, dy = 0, 0
        if pacman.target_x > pacman.x:
            dx = 1
        elif pacman.target_x < pacman.x:
            dx = -1
        elif pacman.target_y > pacman.y:
            dy = 1
        elif pacman.target_y < pacman.y:
            dy = -1

        # 計算圍堵點：Pac-Man 與 Ghost1 中點的延長
        mid_x = (pacman.x + ghost1.x) // 2
        mid_y = (pacman.y + ghost1.y) // 2
        target_x = mid_x + dx * 4
        target_y = mid_y + dy * 4
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))

        # 設置陷阱：選擇路口或關鍵位置
        if ((target_x - pacman.x) ** 2 + (target_y - pacman.y) ** 2) ** 0.5 < 6:
            junctions = [
                (x, y) for x, y in [(pacman.x + dx * 2, pacman.y + dy * 2),
                                    (pacman.x + dy, pacman.y - dx),
                                    (pacman.x - dy, pacman.y + dx)]
                if maze.xy_valid(x, y) and maze.get_tile(x, y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                and sum(maze.xy_valid(x + ddx, y + ddy) and maze.get_tile(x + ddx, y + ddy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                        for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]) > 2
            ]
            if junctions:
                target = min(junctions, key=lambda p: ((p[0] - pacman.x) ** 2 + (p[1] - pacman.y) ** 2) ** 0.5)
                if self.move_to_target(target[0], target[1], maze):
                    return

        if self.move_to_target(target_x, target_y, maze):
            return
        self.move_random(maze)

class Ghost4(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost4"):
        """
        初始化 Ghost4，使用淺藍色，作為攪亂者。
        """
        super().__init__(x, y, name, color=LIGHT_BLUE)
        self.speed = GHOST4_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        攪亂策略：當距離 Pac-Man 小於 6 格時，隨機選擇路口阻塞，否則預測並追逐。
        """
        distance = ((self.x - pacman.x) ** 2 + (self.y - pacman.y) ** 2) ** 0.5
        threshold = 6
        if distance < threshold:
            # 隨機選擇附近路口阻塞
            nearby_points = [
                (self.x + dx, self.y + dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                if maze.xy_valid(self.x + dx, self.y + dy) and
                maze.get_tile(self.x + dx, self.y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
            ]
            junctions = [
                (x, y) for x, y in nearby_points
                if sum(maze.xy_valid(x + ddx, y + ddy) and maze.get_tile(x + ddx, y + ddy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                       for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]) > 1
            ]
            if junctions:
                target = random.choice(junctions)
                if self.move_to_target(target[0], target[1], maze):
                    return
            self.move_random(maze)
            return

        # 預測 Pac-Man 位置並追逐
        dx, dy = 0, 0
        if pacman.target_x != pacman.x or pacman.target_y != pacman.y:
            dx = pacman.target_x - pacman.x
            dy = pacman.target_y - pacman.y
        target_x = pacman.x + dx * 2  # 預測 2 步
        target_y = pacman.y + dy * 2
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        if self.move_to_target(target_x, target_y, maze):
            return
        if self.move_to_target(pacman.x, pacman.y, maze):
            return
        self.move_random(maze)