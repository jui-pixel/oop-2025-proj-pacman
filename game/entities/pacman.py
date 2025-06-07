# game/entities/pacman.py
"""
定義 Pac-Man 實體，包括移動邏輯、得分機制和 AI 路徑規劃。
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
        初始化 Pac-Man，設置初始分數、生命值和生存狀態。
        """
        super().__init__(x, y, 'P')
        self.score = 0
        self.lives = 3  # 初始化 3 條命
        self.alive = True
        self.speed = PACMAN_BASE_SPEED  # 基礎移動速度
        self.last_direction = None  # 記錄上一次移動方向
        self.alternating_vertical_count = 0  # 記錄連續上下交替移動次數
        self.stuck_count = 0  # 連續卡住計數器
        self.max_stuck_frames = MAX_STUCK_FRAMES  # 最大卡住幀數
        self.initial_x = x  # 儲存初始位置
        self.initial_y = y

    def eat_pellet(self, pellets: List['PowerPellet']) -> int:
        """
        檢查並吃能量球，更新分數。
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
        """
        self.lives -= 1
        self.score -= 50  # 每次死亡扣除 50 分
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
        使用 A* 算法找到從 start 到 goal 的最短路徑，考慮迷宮障礙和鬼魂威脅。
        當路徑無效或被鬼擋住時，嘗試替代路徑或安全方向。
        在追逐可食用鬼魂時，避開能量球。
        """
        def heuristic(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        # 預測鬼魂下一個位置
        predicted_danger = set()
        for ghost in ghosts:
            if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                dx = 1 if ghost.x < start[0] else -1 if ghost.x > start[0] else 0
                dy = 1 if ghost.y < start[1] else -1 if ghost.y > start[1] else 0
                next_x, next_y = ghost.x + dx, ghost.y + dy
                if maze.xy_valid(next_x, next_y):
                    predicted_danger.add((next_x, next_y))

        # 逃跑模式下選擇安全區域
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
        f_score = {start: heuristic(start, goal)}

        closed_set = set()

        # 定義威脅區域並加重被鬼擋住的路徑成本
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
                        if dist < 2:  # 靠近鬼魂的格子加重成本
                            danger_cost += 50 / max(1, dist)
                        elif neighbor in predicted_danger:
                            danger_cost += 20
                
                power_avoidance_penalty = 0
                if target_type == "edible" and power_pellets:
                    for pellet in power_pellets:
                        dist = ((neighbor[0] - pellet.x) ** 2 + (neighbor[1] - pellet.y) ** 2) ** 0.5
                        if dist < 2:
                            power_avoidance_penalty += 400 / max(1, dist)

                power_penalty = 0
                if target_type == "score" and power_pellets:
                    for pellet in power_pellets:
                        dist = ((neighbor[0] - pellet.x) ** 2 + (neighbor[1] - pellet.y) ** 2) ** 0.5
                        if dist < 2:
                            power_penalty += 500 / max(1, dist)
                
                score_reward = 0
                if target_type in ["edible", "none", "flee"] and maze.get_tile(neighbor[0], neighbor[1]) != TILE_WALL:
                    for pellet in score_pellets:
                        if pellet.x == neighbor[0] and pellet.y == neighbor[1]:
                            score_reward -= 0.6  # 順路吃掉分數球
                            break
                tentative_g_score = g_score[current] + 1 + danger_cost + power_avoidance_penalty + power_penalty + score_reward

                if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        # 路徑無效時，退回到隨機安全方向
        safe_directions = [(dx, dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] 
                          if maze.xy_valid(start[0] + dx, start[1] + dy) 
                          and maze.get_tile(start[0] + dx, start[1] + dy) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
        return random.choice(safe_directions) if safe_directions else None

    def rule_based_ai_move(self, maze, power_pellets: List['PowerPellet'], score_pellets: List['ScorePellet'], ghosts: List['Ghost']) -> bool:
        """
        智能規則基礎的 AI 移動邏輯，使用 A* 路徑規劃，優先收集分數球，避開活著且不可食用鬼魂的動態威脅區域。
        若無能量球，則盡快吃完分數球。
        若能量球被鬼擋住，動態切換到逃跑或收集分數球。
        在追逐可食用鬼魂時，盡量避開能量球。
        若連續上下交替移動超過 4 次，切換到最近的分數球。
        包含脫困機制以避免卡住。
        
        Args:
            maze (Map): 迷宮物件.
            power_pellets (List[PowerPellet]): 能量球列表.
            score_pellets (List[ScorePellet]): 分數球列表.
            ghosts (List[Ghost]): 鬼魂列表.
        
        Returns:
            bool: 是否成功設置新目標.
        """
        self.speed = PACMAN_AI_SPEED  # 提高速度以適應 AI 移動
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 下、上、右、左
        current_x, current_y = self.x, self.y
        start = (current_x, current_y)

        # 步驟 1: 檢查活著的不可食用鬼魂數量
        alive_ghosts = sum(1 for ghost in ghosts if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible)

        # 步驟 2: 評估威脅 - 檢查最近的危險鬼魂
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

        # 步驟 4: 計算當前移動方向並檢測連續上下交替移動
        direction = None
        if self.last_direction:
            current_dx, current_dy = 0, 0
            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]:
                    if self.set_new_target(dx, dy, maze):
                        current_dx, current_dy = dx, dy
                        direction = (dx, dy)
                        break
            if direction:
                if abs(current_dy) > 0:
                    if self.last_direction and (self.last_direction[1] * current_dy < 0):
                        self.alternating_vertical_count += 1
                        if self.alternating_vertical_count > 4:
                            if score_pellets:
                                score_options = [(s, (s.x - current_x) ** 2 + (s.y - current_y) ** 2) for s in score_pellets]
                                closest_score = min(score_options, key=lambda x: x[1])[0]
                                goal = (closest_score.x, closest_score.y)
                                direction = self.find_path(start, goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score")
                                if direction:
                                    dx, dy = direction
                                    if self.set_new_target(dx, dy, maze):
                                        self.last_direction = (dx, dy)
                                        self.alternating_vertical_count = 0
                                        self.stuck_count = 0
                                        return True
                    else:
                        self.alternating_vertical_count = 0
                else:
                    self.alternating_vertical_count = 0

        # 步驟 5: 模擬幾步路徑選擇
        def simulate_move(dx, dy, steps=3):
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

        # 步驟 6: 如果無能量球，盡快吃完分數球
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

        # 步驟 7: 殞局模式處理
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

        # 步驟 8: 優先級策略
        if min_danger_dist < 6 or alive_ghosts >= 3:
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
            edible_ghosts = [ghost for ghost in ghosts if ghost.edible and ghost.edible_timer > 5 and not ghost.returning_to_spawn and not ghost.waiting]
            if edible_ghosts:
                closest_edible = min(edible_ghosts, key=lambda g: (g.x - current_x) ** 2 + (g.y - current_y) ** 2)
                if ((closest_edible.x - current_x)**2 + (closest_edible.y - current_y)**2) < 144:
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

        # 步驟 9: 選擇最安全方向或脫困
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