# game/entities.py
"""
定義 Pac-Man 遊戲中的實體，包括 Pac-Man、鬼魂、能量球和分數球。
提供移動、碰撞檢測和 AI 移動邏輯。
"""
import random
from typing import Tuple, List
from config import CELL_SIZE

class Entity:
    def __init__(self, x: int, y: int, symbol: str):
        """
        初始化基本實體，設置位置和符號。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            symbol (str): 實體的符號表示。
        """
        self.x = x
        self.y = y
        self.symbol = symbol
        self.target_x = x
        self.target_y = y
        self.current_x = x * CELL_SIZE + CELL_SIZE // 2  # 像素座標
        self.current_y = y * CELL_SIZE + CELL_SIZE // 2
        self.speed = 2.0  # 移動速度（像素/幀）

    def move_towards_target(self, maze) -> bool:
        """
        逐像素移動到目標格子。
        
        Args:
            maze (Map): 迷宮物件。
        
        Returns:
            bool: 是否到達目標格子。
        """
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
        """
        設置新目標格子，檢查是否可通行。
        
        Args:
            dx (int): x 方向偏移。
            dy (int): y 方向偏移。
            maze (Map): 迷宮物件。
        
        Returns:
            bool: 是否成功設置目標。
        """
        new_x, new_y = self.x + dx, self.y + dy
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'E', 'S', 'D']:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

class PacMan(Entity):
    def __init__(self, x: int, y: int):
        """
        初始化 Pac-Man，設置初始分數和生存狀態。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
        """
        super().__init__(x, y, 'P')
        self.score = 0
        self.alive = True
        self.speed = 2.5  # Pac-Man 速度稍快

    def eat_pellet(self, pellets: List['PowerPellet']) -> int:
        """
        檢查並吃能量球，更新分數。
        
        Args:
            pellets (List[PowerPellet]): 能量球列表。
        
        Returns:
            int: 獲得的分數。
        """
        for pellet in pellets[:]:
            if pellet.x == self.x and pellet.y == self.y:
                self.score += pellet.value
                pellets.remove(pellet)
                return pellet.value
        return 0

    def eat_score_pellet(self, score_pellets: List['ScorePellet']) -> int:
        """
        檢查並吃分數球，更新分數。
        
        Args:
            score_pellets (List[ScorePellet]): 分數球列表。
        
        Returns:
            int: 獲得的分數。
        """
        for score_pellet in score_pellets[:]:
            if score_pellet.x == self.x and score_pellet.y == self.y:
                self.score += score_pellet.value
                score_pellets.remove(score_pellet)
                return score_pellet.value
        return 0

    def rule_based_ai_move(self, maze, power_pellets: List['PowerPellet'], score_pellets: List['ScorePellet'], ghosts: List['Ghost']) -> bool:
        """
        規則基礎的 AI 移動邏輯，優先追逐可吃鬼魂或收集最近的球，避免危險鬼魂。
        
        Args:
            maze (Map): 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
        
        Returns:
            bool: 是否成功設置新目標。
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        best_score = -float('inf')
        
        edible_ghosts = [ghost for ghost in ghosts if ghost.edible and ghost.edible_timer > 0]
        if edible_ghosts:
            closest_ghost = min(edible_ghosts, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
            for dx, dy in directions:
                new_x, new_y = self.x + dx, self.y + dy
                if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'A', 'o', 's', 'S']:
                    distance = ((new_x - closest_ghost.x) ** 2 + (new_y - closest_ghost.y) ** 2) ** 0.5
                    score = -distance
                    if score > best_score:
                        best_score = score
                        best_direction = (dx, dy)
        else:
            targets = [(p.x, p.y, 10) for p in power_pellets] + [(s.x, s.y, 2) for s in score_pellets]
            if not targets:
                random.shuffle(directions)
                for dx, dy in directions:
                    if self.set_new_target(dx, dy, maze):
                        return True
                return False
            
            for dx, dy in directions:
                new_x, new_y = self.x + dx, self.y + dy
                if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'A', 'o', 's', 'S']:
                    score = 0
                    for target_x, target_y, value in targets:
                        distance = ((new_x - target_x) ** 2 + (new_y - target_y) ** 2) ** 0.5
                        score += value / max(1, distance)
                    
                    for ghost in ghosts:
                        if not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                            distance = ((new_x - ghost.x) ** 2 + (new_y - ghost.y) ** 2) ** 0.5
                            if distance < 3:
                                score -= 100 / max(1, distance)
                    
                    if score > best_score:
                        best_score = score
                        best_direction = (dx, dy)
        
        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            return True
        return False

class Ghost(Entity):
    def __init__(self, x: int, y: int, name: str, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        初始化鬼魂，設置名稱、顏色和行為屬性。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱。
            color (Tuple[int, int, int]): 鬼魂顏色。
        """
        super().__init__(x, y, 'G')
        self.name = name
        self.color = color
        self.speed = 2.0
        self.edible = False
        self.edible_timer = 0
        self.respawn_timer = 0
        self.returning_to_spawn = False
        self.return_speed = 4.0
        self.death_count = 0
        self.waiting = False
        self.wait_timer = 0
        self.alpha = 255

    def move(self, pacman: PacMan, maze, fps: int):
        """
        根據鬼魂狀態執行移動邏輯（等待、可吃、返回重生點或追逐）。
        
        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
            fps (int): 每秒幀數。
        """
        if self.waiting:
            self.wait_timer -= 1
            if self.wait_timer <= 0:
                print(f"{self.name} finished waiting, resuming chase.")
                self.waiting = False
                self.speed = 2.0
            return
        
        if self.edible:
            self.edible_timer -= 1
            if self.edible_timer <= 0:
                self.edible = False
                self.edible_timer = 0
                print(f"{self.name} is no longer edible.")
        
        if self.respawn_timer > 0:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0 and self.returning_to_spawn:
                self.reset_position(maze, [(x, y) for x, y in [(x, y) for y in range(maze.h) 
                                                               for x in range(maze.w) if maze.get_tile(x, y) == 'S']])
                print(f"{self.name} has respawned at spawn point.")
            return
        
        if self.returning_to_spawn:
            self.return_to_spawn(maze, fps)
        elif self.edible and self.edible_timer > 0:
            self.escape_from_pacman(pacman, maze)
        else:
            self.chase_pacman(pacman, maze)

    def return_to_spawn(self, maze):
        """
        快速返回最近的重生點 'S'。
        
        Args:
            maze (Map): 迷宮物件。
            fps (int): 每秒幀數。
        """
        self.speed = self.return_speed
        spawn_points = [(x, y) for x, y in [(x, y) for y in range(maze.h) 
                                           for x in range(maze.w) if maze.get_tile(x, y) == 'S']]
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
        """
        在可吃狀態下逃離 Pac-Man。
        
        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
        """
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
        self.move_random(maze)

    def move_random(self, maze):
        """
        隨機選擇一個可通行方向移動。
        
        Args:
            maze (Map): 迷宮物件。
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.set_new_target(dx, dy, maze):
                return

    def chase_pacman(self, pacman: PacMan, maze):
        """
        追逐 Pac-Man 的抽象方法，由子類實現。
        
        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
        """
        pass

    def set_edible(self, duration: int):
        """
        設置鬼魂為可吃狀態。
        
        Args:
            duration (int): 可吃持續時間（幀數）。
        """
        self.edible = True
        self.edible_timer = duration
        self.respawn_timer = 0

    def set_returning_to_spawn(self, fps: int):
        """
        設置鬼魂返回重生點。
        
        Args:
            fps (int): 每秒幀數。
        """
        self.death_count += 1
        self.returning_to_spawn = True
        self.edible = False
        self.edible_timer = 0
        self.alpha = 255
        self.respawn_timer = int(5 * fps)

    def set_waiting(self, fps: int):
        """
        設置鬼魂為等待狀態。
        
        Args:
            fps (int): 每秒幀數。
        """
        self.returning_to_spawn = False
        self.waiting = True
        wait_time = 10.0 / max(1, self.death_count)
        self.wait_timer = int(wait_time * fps)

    def reset_position(self, maze, respawn_points):
        """
        重置鬼魂位置到隨機重生點。
        
        Args:
            maze (Map): 迷宮物件。
            respawn_points (List[Tuple]): 重生點座標列表。
        """
        self.edible = False
        self.returning_to_spawn = False
        self.waiting = False
        self.respawn_timer = 0
        self.edible_timer = 0
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
        """
        初始化能量球，設置分數值。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            value (int): 分數值，預設為 10。
        """
        super().__init__(x, y, 'o')
        self.value = value

class ScorePellet(Entity):
    def __init__(self, x: int, y: int, value: int = 2):
        """
        初始化分數球，設置分數值。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            value (int): 分數值，預設為 2。
        """
        super().__init__(x, y, 's')
        self.value = value

from ghosts.ghost1 import Ghost1
from ghosts.ghost2 import Ghost2
from ghosts.ghost3 import Ghost3
from ghosts.ghost4 import Ghost4

def initialize_entities(maze) -> Tuple[PacMan, List[Ghost], List[PowerPellet], List[ScorePellet]]:
    """
    初始化所有遊戲實體，包括 Pac-Man、鬼魂、能量球和分數球。
    
    Args:
        maze (Map): 迷宮物件。
    
    Returns:
        Tuple: (PacMan, List[Ghost], List[PowerPellet], List[ScorePellet])
    """
    import random
    
    # 收集所有可行走的點 ('.')
    valid_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1) 
                      if maze.get_tile(x, y) == '.']
    
    # 定義邊緣與中間之間的區域
    edge_mid_positions = []
    for x, y in valid_positions:
        # 距離邊界 1-2 格，但不完全在中間
        is_near_edge = (x <= 2 or x >= maze.w - 3 or y <= 2 or y >= maze.h - 3)
        is_not_middle = not (maze.w // 4 < x < 3 * maze.w // 4 and maze.h // 4 < y < 3 * maze.h // 4)
        if is_near_edge and is_not_middle:
            edge_mid_positions.append((x, y))
    
    # 從邊緣與中間之間的點中隨機選擇 Pac-Man 出生點
    if edge_mid_positions:
        pacman_pos = random.choice(edge_mid_positions)
    else:
        # 如果沒有符合條件的點，則從所有 '.' 點中隨機選擇
        if valid_positions:
            pacman_pos = random.choice(valid_positions)
        else:
            raise ValueError("No valid start position ('.') for Pac-Man in maze")
    
    pacman = PacMan(pacman_pos[0], pacman_pos[1])
    
    # 初始化鬼魂
    ghosts = []
    ghost_classes = [Ghost1, Ghost2, Ghost3, Ghost4]
    ghost_spawn_points = [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']
    if not ghost_spawn_points:
        raise ValueError("迷宮中沒有'S'格子，無法生成鬼魂！")
    
    random.shuffle(ghost_spawn_points)
    for i, ghost_class in enumerate(ghost_classes):
        spawn_point = ghost_spawn_points[i % len(ghost_spawn_points)]
        ghosts.append(ghost_class(spawn_point[0], spawn_point[1], f"Ghost{i+1}"))
    
    # 初始化能量球
    power_pellets = []
    a_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                   if maze.get_tile(x, y) == 'E' and (x, y) != (pacman.x, pacman.y)]
    for x, y in a_positions:
        power_pellets.append(PowerPellet(x, y))
    
    # 初始化分數球
    all_path_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                         if maze.get_tile(x, y) == '.']
    excluded_positions = [(pacman.x, pacman.y)] + ghost_spawn_points + a_positions
    score_positions = [pos for pos in all_path_positions if pos not in excluded_positions]
    score_pellets = [ScorePellet(x, y) for x, y in score_positions]
    
    return pacman, ghosts, power_pellets, score_pellets