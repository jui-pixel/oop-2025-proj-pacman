# game/entities.py
import random
from typing import Tuple, List

class Entity:
    def __init__(self, x: int, y: int, symbol: str):
        self.x = x
        self.y = y
        self.symbol = symbol  # 角色在迷宮中的表示符號

    def move(self, dx: int, dy: int, maze) -> bool:
        """通用移動方法，檢查新位置是否可通行"""
        new_x, new_y = self.x + dx, self.y + dy
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'A', 'o', 's']:
            self.x, self.y = new_x, new_y
            return True
        return False

class PacMan(Entity):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, 'P')  # 用 'P' 表示 Pac-Man
        self.score = 0
        self.alive = True

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

    def move_random(self, maze) -> None:
        """簡單的隨機移動邏輯（後續由 ghosts/ 模組擴展）"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.move(dx, dy, maze):
                break

class PowerPellet(Entity):
    def __init__(self, x: int, y: int, value: int = 10):
        super().__init__(x, y, 'o')  # 用 'o' 表示能量球
        self.value = value  # 吃到能量球的得分

class ScorePellet(Entity):
    def __init__(self, x: int, y: int, value: int = 2):
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
    # 計算中央房間的範圍（7x5 房間，根據 maze_generator.py 中的 _add_central_room）
    room_w, room_h = 7, 5
    start_x = (maze.w - room_w) // 2
    start_y = (maze.h - room_h) // 2
    room_area = [(x, y) for y in range(start_y, start_y + room_h) for x in range(start_x, start_x + room_w)]

    # 找到所有 '.' 點，但排除中央房間、Pac-Man 位置、'S' 位置和能量球位置
    all_path_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                         if maze.get_tile(x, y) == '.']
    excluded_positions = room_area + [(pacman.x, pacman.y)] + [(x, y) for (x, y) in a_positions] + \
                        [(x, y) for (x, y) in [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']]
    score_positions = [pos for pos in all_path_positions if pos not in excluded_positions]

    # 放置分數球
    score_pellets = []
    for x, y in score_positions:
        score_pellets.append(ScorePellet(x, y))

    return pacman, ghosts, power_pellets, score_pellets