# game/entities.py
import random
from typing import Tuple, List

CELL_SIZE = 20  # 每個格子的大小（像素）

class Entity:
    def __init__(self, x: int, y: int, symbol: str):
        self.x = x
        self.y = y
        self.symbol = symbol
        self.target_x = x
        self.target_y = y
        self.current_x = x * CELL_SIZE + CELL_SIZE // 2
        self.current_y = y * CELL_SIZE + CELL_SIZE // 2
        self.speed = 2.0

    def move_towards_target(self, maze):
        """逐像素移動到目標格子"""
        target_pixel_x = self.target_x * CELL_SIZE + CELL_SIZE // 2
        target_pixel_y = self.target_y * CELL_SIZE + CELL_SIZE // 2
        dx = target_pixel_x - self.current_x
        dy = target_pixel_y - self.current_y
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if dist <= self.speed:
            self.current_x = target_pixel_x
            self.current_y = target_pixel_y
            self.x = self.target_x
            self.y = self.target_y
            return True
        else:
            if dist != 0:
                self.current_x += (dx / dist) * self.speed
                self.current_y += (dy / dist) * self.speed
            return False

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """檢查新目標是否可通行，若可則設置新目標"""
        new_x, new_y = self.x + dx, self.y + dy
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'A', 'o', 's', 'S']:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

class PacMan(Entity):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, 'P')
        self.score = 0
        self.alive = True
        self.speed = 2.5

    def eat_pellet(self, pellets: List['PowerPellet']) -> int:
        for pellet in pellets:
            if pellet.x == self.x and pellet.y == self.y:
                self.score += pellet.value
                pellets.remove(pellet)
                return pellet.value
        return 0

    def eat_score_pellet(self, score_pellets: List['ScorePellet']) -> int:
        for score_pellet in score_pellets:
            if score_pellet.x == self.x and score_pellet.y == self.y:
                self.score += score_pellet.value
                score_pellets.remove(score_pellet)
                return score_pellet.value
        return 0

    def check_collision(self, ghosts: List['Ghost']) -> bool:
        for ghost in ghosts:
            if ghost.x == self.x and ghost.y == self.y:
                self.alive = False
                return True
        return False

class Ghost(Entity):
    def __init__(self, x: int, y: int, name: str):
        super().__init__(x, y, 'G')
        self.name = name
        self.speed = 2.0
        self.edible = False
        self.respawn_timer = 0
        self.returning_to_spawn = False
        self.return_speed = 4.0
        self.death_count = 0
        self.waiting = False
        self.wait_timer = 0
        self.alpha = 255

    def move(self, pacman: PacMan, maze, fps: int):
        """根據當前狀態決定移動行為"""
        if self.waiting:
            self.wait_timer -= 1
            if self.wait_timer <= 0:
                print(f"{self.name} finished waiting, resuming chase.")
                self.waiting = False
                self.speed = 2.0
            return
        if self.respawn_timer > 0:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                print(f"{self.name} has respawned.")
                self.reset_position(maze, [(x, y) for x, y in [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']])
            return
        if self.returning_to_spawn:
            self.return_to_spawn(maze, fps)
        elif self.edible:
            self.escape_from_pacman(pacman, maze)  # 可被吃時逃離玩家
        else:
            self.chase_pacman(pacman, maze)

    def return_to_spawn(self, maze, fps: int):
        """快速返回最近的 'S' 點"""
        self.speed = self.return_speed
        spawn_points = [(x, y) for x, y in [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']]
        if not spawn_points:
            self.move_random(maze)
            return

        closest_spawn = min(spawn_points, key=lambda p: (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2)
        dx = 1 if closest_spawn[0] > self.x else -1 if closest_spawn[0] < self.x else 0
        dy = 1 if closest_spawn[1] > self.y else -1 if closest_spawn[1] < self.y else 0

        if dx != 0 and self.set_new_target(dx, 0, maze):
            return
        if dy != 0 and self.set_new_target(0, dy, maze):
            return
        self.move_random(maze)

    def escape_from_pacman(self, pacman: PacMan, maze):
        """逃離 Pac-Man，選擇與玩家距離最遠的方向"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        max_distance = -1

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'A', 'o', 's', 'S']:
                distance = ((new_x - pacman.x) ** 2 + (new_y - pacman.y) ** 2) ** 0.5
                if distance > max_distance:
                    max_distance = distance
                    best_direction = (dx, dy)

        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            return
        self.move_random(maze)  # 如果無法遠離，隨機移動

    def move_random(self, maze):
        """隨機選擇一個方向，並設置新目標"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.set_new_target(dx, dy, maze):
                return

    def chase_pacman(self, pacman: PacMan, maze):
        """追逐 Pac-Man 的抽象方法，必須由子類實現"""
        pass

    def set_edible(self, duration: int):
        self.edible = True
        self.respawn_timer = duration

    def set_returning_to_spawn(self, fps: int):
        self.death_count += 1
        self.returning_to_spawn = True
        self.edible = False
        self.alpha = 255

    def set_waiting(self, fps: int):
        self.returning_to_spawn = False
        self.waiting = True
        wait_time = 10.0 / max(1, self.death_count)
        self.wait_timer = int(wait_time * fps)
        print(f"{self.name} is waiting for {wait_time} seconds (death count: {self.death_count}).")

    def reset_position(self, maze, respawn_points):
        self.edible = False
        self.returning_to_spawn = False
        self.waiting = False
        self.respawn_timer = 0
        self.speed = 2.0
        self.alpha = 255
        if respawn_points:
            spawn_point = random.choice(respawn_points)
            self.x, self.y = spawn_point
            self.target_x, self.target_y = spawn_point
            self.current_x = self.x * CELL_SIZE + CELL_SIZE // 2
            self.current_y = self.y * CELL_SIZE + CELL_SIZE // 2

class PowerPellet(Entity):
    def __init__(self, x: int, y: int, value: int = 10):
        super().__init__(x, y, 'o')
        self.value = value

class ScorePellet(Entity):
    def __init__(self, x: int, y: int, value: int = 2):
        super().__init__(x, y, 's')
        self.value = value

from ghosts.ghost1 import Ghost1
from ghosts.ghost2 import Ghost2
from ghosts.ghost3 import Ghost3
from ghosts.ghost4 import Ghost4

def initialize_entities(maze) -> Tuple[PacMan, List[Ghost], List[PowerPellet], List[ScorePellet]]:
    pacman = None
    for y in range(1, maze.h - 1):
        for x in range(1, maze.w - 1):
            if maze.get_tile(x, y) == '.':
                pacman = PacMan(x, y)
                break
        if pacman:
            break

    ghosts = []
    ghost_classes = [Ghost1, Ghost2, Ghost3, Ghost4]
    ghost_spawn_points = [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']
    if not ghost_spawn_points:
        raise ValueError("迷宮中沒有'S'格子，無法生成鬼魂！")

    random.shuffle(ghost_spawn_points)
    for i, ghost_class in enumerate(ghost_classes):
        spawn_point = ghost_spawn_points[i % len(ghost_spawn_points)]
        ghosts.append(ghost_class(spawn_point[0], spawn_point[1], f"Ghost{i+1}"))

    power_pellets = []
    a_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                   if maze.get_tile(x, y) == 'A' and (x, y) != (pacman.x, pacman.y)]
    for x, y in a_positions:
        power_pellets.append(PowerPellet(x, y))

    all_path_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                         if maze.get_tile(x, y) == '.']
    excluded_positions = [(pacman.x, pacman.y)] + ghost_spawn_points + a_positions
    score_positions = [pos for pos in all_path_positions if pos not in excluded_positions]
    score_pellets = []
    for x, y in score_positions:
        score_pellets.append(ScorePellet(x, y))

    return pacman, ghosts, power_pellets, score_pellets