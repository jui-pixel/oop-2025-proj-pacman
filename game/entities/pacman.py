# game/entities/pacman.py
"""
定義 Pac-Man 實體，包括移動邏輯、得分機制和基於 A* 算法的規則基礎 AI 路徑規劃。
"""

from .entity_base import Entity
from heapq import heappush, heappop
from typing import Tuple, List
import random
from config import CELL_SIZE, TILE_BOUNDARY, TILE_WALL, TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR, PACMAN_BASE_SPEED, PACMAN_AI_SPEED, MAX_STUCK_FRAMES
from .pellets import PowerPellet, ScorePellet
from .ghost import Ghost

class PacMan(Entity):
    def __init__(self, x: int, y: int):
        """
        初始化 Pac-Man，設置初始分數、生命值和狀態屬性。

        原理：
        - Pac-Man 是遊戲的主角，具有生命值、得分和移動狀態。
        - 初始設置 3 條命，分數為 0，支持 AI 控制的移動邏輯。
        - 狀態屬性包括：
          - score：當前分數。
          - lives：剩餘生命數。
          - last_direction：上一次移動方向，用於檢測反覆移動。
          - stuck_count：卡住計數器，用於脫困。

        Args:
            x (int): 迷宮中的 x 坐標（格子坐標）。
            y (int): 迷宮中的 y 坐標（格子坐標）。
        """
        super().__init__(x, y, 'P')  # 調用基類 Entity 初始化
        self.score = 0  # 初始分數
        self.lives = 3  # 初始生命數
        self.alive = True  # 生存狀態
        self.speed = PACMAN_BASE_SPEED  # 基礎移動速度
        self.last_direction = None  # 上次移動方向
        self.alternating_vertical_count = 0  # 連續上下交替移動次數
        self.stuck_count = 0  # 連續卡住計數
        self.max_stuck_frames = MAX_STUCK_FRAMES  # 最大卡住幀數
        self.initial_x = x  # 初始 x 坐標
        self.initial_y = y  # 初始 y 坐標

    def eat_pellet(self, pellets: List['PowerPellet']) -> int:
        """
        檢查並吃能量球，更新分數並移除能量球。

        原理：
        - 若 Pac-Man 位置與能量球重合，增加分數並移除該能量球。
        - 分數增量等於能量球的 value（預設 10）。

        Args:
            pellets (List[PowerPellet]): 能量球列表。

        Returns:
            int: 增加的分數（若未吃到則為 0）。
        """
        for pellet in pellets[:]:
            if pellet.x == self.x and pellet.y == self.y:
                self.score += pellet.value
                pellets.remove(pellet)
                return pellet.value
        return 0

    def eat_score_pellet(self, score_pellets: List['ScorePellet']) -> int:
        """
        檢查並吃分數球，更新分數並移除分數球。

        原理：
        - 若 Pac-Man 位置與分數球重合，增加分數並移除該分數球。
        - 分數增量等於分數球的 value（預設 2）。

        Args:
            score_pellets (List[ScorePellet]): 分數球列表。

        Returns:
            int: 增加的分數（若未吃到則為 0）。
        """
        for score_pellet in score_pellets[:]:
            if score_pellet.x == self.x and score_pellet.y == self.y:
                self.score += score_pellet.value
                score_pellets.remove(score_pellet)
                return score_pellet.value
        return 0
    
    def lose_life(self, maze) -> None:
        """
        扣除一條命並重置 Pac-Man 位置到初始位置。

        原理：
        - 當 Pac-Man 被不可食用鬼魂碰到時，損失一條命並扣除 50 分。
        - 位置重置到初始坐標 (initial_x, initial_y)，並清空移動狀態。
        - 若生命數為 0，遊戲結束（由環境處理）。

        Args:
            maze: 迷宮物件。
        """
        self.lives -= 1
        self.score -= 50  # 死亡扣分
        # self.x = self.initial_x
        # self.y = self.initial_y
        # self.current_x = self.x * CELL_SIZE + CELL_SIZE // 2
        # self.current_y = self.y * CELL_SIZE + CELL_SIZE // 2
        # self.target_x = self.x
        # self.target_y = self.y
        # self.last_direction = None
        # self.alternating_vertical_count = 0
        # self.stuck_count = 0

    def find_path(self, start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score"):
        """
        使用 A* 算法找到從 start 到 goal 的最短路徑，考慮鬼魂威脅和目標類型。

        原理：
        - A* 算法結合啟發式函數（歐幾里得距離）尋找最短路徑，考慮迷宮障礙和鬼魂威脅。
        - 啟發式函數：h(a, b) = √((a[0] - b[0])^2 + (a[1] - b[1])^2)
        - 成本函數 g 包含：
          - 基本移動成本（1）。
          - 鬼魂威脅成本（danger_cost，與距離成反比）。
          - 能量球懲罰（power_penalty，target_type="score" 時避免能量球）。
          - 分數球獎勵（score_reward，順路吃分數球）。
        - 模式：
          - approach：直接接近目標。
          - flee：逃離危險鬼魂，選擇最遠的安全點。
        - 目標類型：score（分數球）、power（能量球）、edible（可食用鬼魂）、none（無目標）。

        Args:
            start (Tuple[int, int]): 起始位置 (x, y)。
            goal (Tuple[int, int]): 目標位置 (x, y)，若為 None 則動態選擇。
            maze: 迷宮物件。
            ghosts (List[Ghost]): 鬼魂列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            power_pellets (List[PowerPellet]): 能量球列表。
            mode (str): 移動模式（"approach" 或 "flee"）。
            target_type (str): 目標類型（"score", "power", "edible", "none"）。

        Returns:
            Tuple[int, int]: 第一步方向 (dx, dy)，或 None。
        """
        def heuristic(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5  # 歐幾里得距離

        # 預測鬼魂下一個位置
        predicted_danger = set()
        for ghost in ghosts:
            if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                dx = 1 if ghost.x < start[0] else -1 if ghost.x > start[0] else 0
                dy = 1 if ghost.y < start[1] else -1 if ghost.y > start[1] else 0
                next_x, next_y = ghost.x + dx, ghost.y + dy
                if maze.xy_valid(next_x, next_y):
                    predicted_danger.add((next_x, next_y))

        # 逃跑模式：選擇遠離危險鬼魂的安全點
        if mode == "flee" and goal is None:
            danger_ghosts = [ghost for ghost in ghosts if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible]
            if not danger_ghosts:
                return None
            best_goal = None
            max_dist = -float('inf')
            center_x, center_y = start[0], start[1]
            search_radius = min(maze.width, maze.height) // 2
            for y in range(max(0, center_y - search_radius), min(maze.height, center_y + search_radius + 1)):
                for x in range(max(0, center_x - search_radius), min(maze.width, center_x + search_radius + 1)):
                    if maze.get_tile(x, y) not in [TILE_BOUNDARY, TILE_WALL] and (x, y) not in predicted_danger:
                        min_ghost_dist = min(((x - ghost.x) ** 2 + (y - ghost.y) ** 2) ** 0.5 for ghost in danger_ghosts)
                        if min_ghost_dist > max_dist:
                            max_dist = min_ghost_dist
                            best_goal = (x, y)
            goal = best_goal if best_goal else start

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}  # 初始啟發值

        closed_set = set()

        # 定義威脅區域
        danger_obstacles = set()
        for ghost in ghosts:
            if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                threat_radius = 1 + int(min(2, max(0, 6 - ((start[0] - ghost.x) ** 2 + (start[1] - ghost.y) ** 2) ** 0.5)))
                for dx in range(-threat_radius, threat_radius + 1):
                    for dy in range(-threat_radius, threat_radius + 1):
                        x, y = ghost.x + dx, ghost.y + dy
                        if maze.xy_valid(x, y):
                            danger_obstacles.add((x, y))

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                if path:
                    next_pos = path[0]
                    dx, dy = next_pos[0] - start[0], next_pos[1] - start[1]
                    return (dx, dy)
                return None

            closed_set.add(current)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not maze.xy_valid(neighbor[0], neighbor[1]) or maze.get_tile(neighbor[0], neighbor[1]) in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]:
                    continue

                danger_cost = 0
                for ghost in ghosts:
                    if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                        dist = ((neighbor[0] - ghost.x) ** 2 + (neighbor[1] - ghost.y) ** 2) ** 0.5
                        if dist < 3:
                            danger_cost += 1000 / max(1, dist)  # 靠近鬼魂加重成本
                        elif neighbor in predicted_danger:
                            danger_cost += 2000  # 預測鬼魂位置加重成本
                
                power_avoidance_penalty = 0
                if target_type == "edible" and power_pellets:
                    for pellet in power_pellets:
                        dist = ((neighbor[0] - pellet.x) ** 2 + (neighbor[1] - pellet.y) ** 2) ** 0.5
                        if dist < 2:
                            power_avoidance_penalty += 100 / max(1, dist)  # 避免能量球

                power_penalty = 0
                if target_type == "score" and power_pellets:
                    for pellet in power_pellets:
                        dist = (neighbor[0] == pellet.x) and (neighbor[1] == pellet.y)
                        if dist:
                            power_penalty += 1000  # 避免能量球

                score_reward = 0
                if target_type in ["edible", "none", "flee"] and maze.get_tile(neighbor[0], neighbor[1]) != TILE_WALL:
                    for pellet in score_pellets:
                        if pellet.x == neighbor[0] and pellet.y == neighbor[1]:
                            score_reward -= 0.9  # 順路吃分數球

                tentative_g_score = g_score[current] + 1 + danger_cost + power_avoidance_penalty + power_penalty + score_reward

                if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        # 若無路徑，選擇隨機安全方向
        safe_directions = [(dx, dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] 
                          if maze.xy_valid(start[0] + dx, start[1] + dy) 
                          and maze.get_tile(start[0] + dx, start[1] + dy) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
        return random.choice(safe_directions) if safe_directions else None

    def rule_based_ai_move(self, maze, power_pellets: List['PowerPellet'], score_pellets: List['ScorePellet'], ghosts: List['Ghost']) -> bool:
        """
        智能規則基礎的 AI 移動邏輯，使用 A* 路徑規劃，優先收集分數球，避開危險鬼魂。

        原理：
        - 採用多層決策邏輯，根據遊戲狀態選擇最佳行動：
          1. 檢查危險鬼魂（不可食用且非返回/等待狀態）的距離。
          2. 若無能量球，優先吃分數球。
          3. 在殞局模式（分數球 ≤ 10）下，動態選擇能量球或分數球。
          4. 若有威脅（距離 < 6），優先吃能量球或逃跑。
          5. 若有可食用鬼魂，追逐最近的鬼魂（距離 < 100）。
          6. 否則，接近最近的分數球。
        - 使用 A* 算法計算路徑，考慮鬼魂威脅和能量球懲罰。
        - 包含脫困機制：若卡住超過 MAX_STUCK_FRAMES，隨機選擇安全方向。

        Args:
            maze: 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。

        Returns:
            bool: 是否成功設置新目標。
        """
        self.speed = PACMAN_AI_SPEED  # 使用 AI 專用速度
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 下、上、右、左
        current_x, current_y = self.x, self.y
        start = (current_x, current_y)

        # 步驟 1: 檢查活著的不可食用鬼魂數量
        alive_ghosts = sum(1 for ghost in ghosts if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible)

        # 步驟 2: 評估威脅
        min_danger_dist = float('inf')
        nearest_danger = None
        for ghost in ghosts:
            if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                dist = ((current_x - ghost.x) ** 2 + (current_y - ghost.y) ** 2) ** 0.5
                if dist < min_danger_dist:
                    min_danger_dist = dist
                    nearest_danger = ghost

        # 步驟 3: 檢查是否進入殞局模式
        is_endgame = len(score_pellets) <= 10

        # 步驟 4: 模擬幾步路徑選擇
        def simulate_move(dx, dy, steps=3):
            """
            模擬移動幾步，檢查安全性。

            原理：
            - 模擬 Pac-Man 沿指定方向移動 steps 步，檢查是否進入危險區域。
            - 若接近鬼魂（距離 < 2）或撞牆，返回無窮大成本。
            - 返回移動距離：dist = √((x - current_x)^2 + (y - current_y)^2)

            Args:
                dx (int): x 方向偏移。
                dy (int): y 方向偏移。
                steps (int): 模擬步數。

            Returns:
                float: 模擬移動的成本。
            """
            x, y = current_x, current_y
            for _ in range(steps):
                x, y = x + dx, y + dy
                if not maze.xy_valid(x, y) or maze.get_tile(x, y) in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]:
                    return float('inf')
                for ghost in ghosts:
                    if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                        dist = ((x - ghost.x) ** 2 + (y - ghost.y) ** 2) ** 0.5
                        if dist < 2:
                            return float('inf')
            return ((x - current_x) ** 2 + (y - current_y) ** 2) ** 0.5

        # 步驟 5: 若無能量球，優先吃分數球
        if not power_pellets and score_pellets:
            immediate_threat = min_danger_dist < 2
            if immediate_threat:
                direction = self.find_path(start, None, maze, ghosts, score_pellets, power_pellets, mode="flee", target_type="none")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True
            else:
                score_options = [(s, (s.x - current_x) ** 2 + (s.y - current_y) ** 2) for s in score_pellets]
                closest_score = min(score_options, key=lambda x: x[1])[0]
                goal = (closest_score.x, closest_score.y)
                direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True

        # 步驟 6: 殞局模式處理
        if is_endgame and score_pellets:
            remaining_scores = [(s, (s.x, s.y)) for s in score_pellets]
            current_pos = start
            best_path = []
            while remaining_scores:
                closest = min(remaining_scores, key=lambda s: ((s[1][0] - current_pos[0]) ** 2 + (s[1][1] - current_pos[1]) ** 2) ** 0.5)
                best_path.append(closest[1])
                current_pos = closest[1]
                remaining_scores.remove(closest)

            use_power_pellet = False
            if power_pellets and (min_danger_dist < 6 or alive_ghosts >= 3):
                power_options = [(p, (p.x - current_x) ** 2 + (p.y - current_y) ** 2) for p in power_pellets]
                closest_power = min(power_options, key=lambda x: x[1])[0]
                power_dist = ((closest_power.x - current_x) ** 2 + (closest_power.y - current_y) ** 2) ** 0.5
                if power_dist < sum(((p[0] - current_x) ** 2 + (p[1] - current_y) ** 2) ** 0.5 for p in best_path) / len(best_path):
                    use_power_pellet = True
                    goal = (closest_power.x, closest_power.y)
                    direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="power")
                    if direction:
                        dx, dy = direction
                        if self.set_new_target(dx, dy, maze):
                            self.last_direction = (dx, dy)
                            self.stuck_count = 0
                            return True

            if not use_power_pellet and best_path:
                goal = best_path[0]
                direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True

        # 步驟 7: 優先級策略
        if min_danger_dist < 6:
            if power_pellets:
                power_options = [(p, (p.x - current_x) ** 2 + (p.y - current_y) ** 2) for p in power_pellets]
                if power_options:
                    closest_power = min(power_options, key=lambda x: x[1])[0]
                    goal = (closest_power.x, closest_power.y)
                    direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="power")
                    if direction:
                        dx, dy = direction
                        if self.set_new_target(dx, dy, maze):
                            self.last_direction = (dx, dy)
                            self.stuck_count = 0
                            return True
                    else:
                        if score_pellets:
                            score_options = [(s, (s.x - current_x) ** 2 + (s.y - current_y) ** 2) for s in score_pellets]
                            closest_score = min(score_options, key=lambda x: x[1])[0]
                            goal = (closest_score.x, closest_score.y)
                            direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score")
                            if direction:
                                dx, dy = direction
                                if self.set_new_target(dx, dy, maze):
                                    self.last_direction = (dx, dy)
                                    self.stuck_count = 0
                                    return True
                        direction = self.find_path(start, None, maze, ghosts, score_pellets, power_pellets, mode="flee", target_type="none")
                        if direction:
                            dx, dy = direction
                            if self.set_new_target(dx, dy, maze):
                                self.last_direction = (dx, dy)
                                self.stuck_count = 0
                                return True
            else:
                direction = self.find_path(start, None, maze, ghosts, score_pellets, power_pellets, mode="flee", target_type="none")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True
        else:
            edible_ghosts = [ghost for ghost in ghosts if ghost.edible and ghost.edible_timer > 3 and not ghost.returning_to_spawn and not ghost.waiting]
            if edible_ghosts:
                closest_edible = min(edible_ghosts, key=lambda g: (g.x - current_x) ** 2 + (g.y - current_y) ** 2)
                if ((closest_edible.x - current_x)**2 + (closest_edible.y - current_y)**2) < 100:
                    goal = (closest_edible.x, closest_edible.y)
                    direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="edible")
                    if direction:
                        dx, dy = direction
                        if self.set_new_target(dx, dy, maze):
                            self.last_direction = (dx, dy)
                            self.stuck_count = 0
                            return True
            
            if score_pellets:
                score_options = [(s, (s.x - current_x) ** 2 + (s.y - current_y) ** 2) for s in score_pellets]
                closest_score = min(score_options, key=lambda x: x[1])[0]
                goal = (closest_score.x, closest_score.y)
                direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True
            else:
                if edible_ghosts:
                    closest_edible = min(edible_ghosts, key=lambda g: (g.x - current_x) ** 2 + (g.y - current_y) ** 2)
                    goal = (closest_edible.x, closest_edible.y)
                    direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="edible")
                    if direction:
                        dx, dy = direction
                        if self.set_new_target(dx, dy, maze):
                            self.last_direction = (dx, dy)
                            self.stuck_count = 0
                            return True

        # 步驟 8: 選擇最安全方向或脫困
        safe_directions = [d for d in directions if maze.xy_valid(current_x + d[0], current_y + d[1]) 
                         and maze.get_tile(current_x + d[0], current_y + d[1]) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
        if not safe_directions:
            return False

        best_score = float('inf')
        best_direction = None
        for dx, dy in safe_directions:
            score = simulate_move(dx, dy)
            if score < best_score:
                best_score = score
                best_direction = (dx, dy)

        if best_direction:
            dx, dy = best_direction
            if self.set_new_target(dx, dy, maze):
                self.last_direction = (dx, dy)
                self.stuck_count = 0
                return True
        else:
            self.stuck_count += 1
            if self.stuck_count > self.max_stuck_frames:
                random_direction = random.choice(safe_directions)
                dx, dy = random_direction
                if self.set_new_target(dx, dy, maze):
                    self.last_direction = (dx, dy)
                    self.stuck_count = 0
                    return True

        return False