# game/entities.py
import random
from typing import Tuple, List

# 為了避免 main.py 中的 CELL_SIZE 引用問題，這裡臨時定義一個常量
# 後續建議將其移動到 settings.py 中
CELL_SIZE = 20  # 每個格子的大小（像素）

class Entity:
    def __init__(self, x: int, y: int, symbol: str):
        self.x = x  # 當前格子 x 座標（整數）
        self.y = y  # 當前格子 y 座標（整數）
        self.symbol = symbol  # 角色在迷宮中的表示符號
        self.target_x = x  # 目標格子的 x 座標
        self.target_y = y  # 目標格子的 y 座標
        self.current_x = x * CELL_SIZE + CELL_SIZE // 2  # 當前像素位置（中心點）
        self.current_y = y * CELL_SIZE + CELL_SIZE // 2  # 當前像素位置（中心點）
        self.speed = 2.0  # 每幀移動的像素數

    def move_towards_target(self, maze):
        """逐像素移動到目標格子"""
        target_pixel_x = self.target_x * CELL_SIZE + CELL_SIZE // 2
        target_pixel_y = self.target_y * CELL_SIZE + CELL_SIZE // 2
        dx = target_pixel_x - self.current_x
        dy = target_pixel_y - self.current_y
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if dist <= self.speed:
            # 已到達目標格子，更新格子座標
            self.current_x = target_pixel_x
            self.current_y = target_pixel_y
            self.x = self.target_x
            self.y = self.target_y
            return True  # 表示已到達目標，可以選擇新目標
        else:
            # 按速度逐像素移動
            angle = (dx ** 2 + dy ** 2) ** 0.5
            if angle != 0:  # 避免除以 0
                self.current_x += (dx / angle) * self.speed
                self.current_y += (dy / angle) * self.speed
            return False  # 表示尚未到達目標

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """檢查新目標是否可通行，若可則設置新目標"""
        new_x, new_y = self.x + dx, self.y + dy
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'A', 'o', 's']:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

class PacMan(Entity):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, 'P')  # 用 'P' 表示 Pac-Man
        self.score = 0
        self.alive = True
        self.speed = 2.5  # Pac-Man 稍快一點

    def eat_pellet(self, pellets: List['PowerPellet']) -> int:
        """檢查是否吃到能量球，並更新分數"""
        for pellet in pellets:
            if pellet.x == self.x and pellet.y == self.y:
                self.score += pellet.value
                pellets.remove(pellet)
                return pellet.value
        return 0

    def eat_score_pellet(self, score_pellets: List['ScorePellet']) -> int:
        """檢查是否吃到分數球，並更新分數"""
        for score_pellet in score_pellets:
            if score_pellet.x == self.x and score_pellet.y == self.y:
                self.score += score_pellet.value
                score_pellets.remove(score_pellet)
                return score_pellet.value
        return 0

    def check_collision(self, ghosts: List['Ghost']) -> bool:
        """檢查是否與鬼魂碰撞，若是則死亡"""
        for ghost in ghosts:
            if ghost.x == self.x and ghost.y == self.y:
                self.alive = False
                return True
        return False

class Ghost(Entity):
    def __init__(self, x: int, y: int, name: str):
        super().__init__(x, y, 'G')  # 用 'G' 表示鬼魂
        self.name = name
        self.speed = 2.0  # 鬼魂速度

    def move_random(self, maze):
        """隨機選擇一個方向，並設置新目標"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.set_new_target(dx, dy, maze):
                break

class PowerPellet(Entity):
    def __init__(self, x: int, y: int, value: int = 10):
        super().__init__(x, y, 'o')  # 用 'o' 表示能量球
        self.value = value  # 吃到能量球的得分

class ScorePellet(Entity):
    def __init__(self, x: int, y: int, value: int = 5):
        super().__init__(x, y, 's')  # 用 's' 表示分數球
        self.value = value  # 吃到分數球的得分

def initialize_entities(maze) -> Tuple[PacMan, List[Ghost], List[PowerPellet], List[ScorePellet]]:
    """初始化遊戲角色"""
    # 初始化 Pac-Man，放置在迷宮左上角第一個可通行位置
    pacman = None
    for y in range(1, maze.h - 1):
        for x in range(1, maze.w - 1):
            if maze.get_tile(x, y) == '.':
                pacman = PacMan(x, y)
                break
        if pacman:
            break

    # 初始化鬼魂，從 'S' 位置生成
    ghosts = []
    for y in range(maze.h):
        for x in range(maze.w):
            if maze.get_tile(x, y) == 'S':
                ghosts.append(Ghost(x, y, f"Ghost-{len(ghosts)+1}"))

    # 放置能量球（PowerPellet），在所有 'A' 點
    power_pellets = []
    a_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                   if maze.get_tile(x, y) == 'A' and (x, y) != (pacman.x, pacman.y)]
    for x, y in a_positions:
        power_pellets.append(PowerPellet(x, y))

    # 放置分數球（ScorePellet），在中央房間和能量球外的所有 '.' 路徑
    room_w, room_h = 7, 5
    start_x = (maze.w - room_w) // 2
    start_y = (maze.h - room_h) // 2
    room_area = [(x, y) for y in range(start_y, start_y + room_h) for x in range(start_x, start_x + room_w)]
    all_path_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                         if maze.get_tile(x, y) == '.']
    excluded_positions = room_area + [(pacman.x, pacman.y)] + [(x, y) for (x, y) in a_positions] + \
                        [(x, y) for (x, y) in [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']]
    score_positions = [pos for pos in all_path_positions if pos not in excluded_positions]
    score_pellets = []
    for x, y in score_positions:
        score_pellets.append(ScorePellet(x, y))

    return pacman, ghosts, power_pellets, score_pellets