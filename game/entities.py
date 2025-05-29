"""
定義 Pac-Man 遊戲中的實體，包括 Pac-Man、鬼魂、能量球和分數球。
提供移動、碰撞檢測和 AI 移動邏輯。
"""
import random
from typing import Tuple, List
from config import CELL_SIZE
from heapq import heappush, heappop

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
        逐像素移動到目標格子，防止速度溢出。
        
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
                move_dist = min(self.speed, dist)  # 限制單幀移動距離
                self.current_x += (dx / dist) * move_dist
                self.current_y += (dy / dist) * move_dist
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
        self.speed = 3.0  # 基礎移動速度
        self.last_direction = None  # 記錄上一次移動方向
        self.alternating_vertical_count = 0  # 記錄連續上下交替移動次數
        self.stuck_count = 0  # 連續卡住計數器
        self.max_stuck_frames = 10  # 最大卡住幀數

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
    
    def find_path(self, start, goal, maze, ghosts, power_pellets, mode="approach", target_type="score"):
        """
        使用 A* 算法找到從 start 到 goal 的最短路徑，考慮迷宮障礙和鬼魂威脅。
        當路徑無效時，退回到隨機安全方向。
        
        Args:
            start (tuple): 起點 (x, y)。
            goal (tuple): 目標 (x, y)，在逃跑模式下可為 None。
            maze (Map): 迷宮物件。
            ghosts (List[Ghost]): 鬼魂列表。
            power_pellets (List[PowerPellet]): 能量球列表。
            mode (str): "approach"（接近目標）或 "flee"（逃離危險）。
            target_type (str): 目標類型，"score"（分數球）、"power"（能量球）、"edible"（可食用鬼魂）。

        Returns:
            tuple: 第一步的方向 (dx, dy)，如果無路徑則返回隨機安全方向。
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
            search_radius = min(maze.w, maze.h) // 2
            for y in range(max(0, center_y - search_radius), min(maze.h, center_y + search_radius + 1)):
                for x in range(max(0, center_x - search_radius), min(maze.w, center_x + search_radius + 1)):
                    if maze.get_tile(x, y) not in ['#', 'X'] and (x, y) not in predicted_danger:
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
                if not maze.xy_valid(neighbor[0], neighbor[1]) or maze.get_tile(neighbor[0], neighbor[1]) in ['#', 'X']:
                    continue

                danger = False
                for ghost in ghosts:
                    if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                        dist = ((neighbor[0] - ghost.x) ** 2 + (neighbor[1] - ghost.y) ** 2) ** 0.5
                        if dist < 2:
                            danger = True
                            break
                if danger and neighbor in predicted_danger:
                    continue

                power_penalty = 0
                if target_type == "score" and power_pellets:
                    for pellet in power_pellets:
                        dist = ((neighbor[0] - pellet.x) ** 2 + (neighbor[1] - pellet.y) ** 2) ** 0.5
                        if dist < 2:
                            power_penalty += 100 / max(1, dist)

                tentative_g_score = g_score[current] + 1 + power_penalty

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
                          and maze.get_tile(start[0] + dx, start[1] + dy) not in ['#', 'X']]
        return random.choice(safe_directions) if safe_directions else None

    def rule_based_ai_move(self, maze, power_pellets: List['PowerPellet'], score_pellets: List['ScorePellet'], ghosts: List['Ghost']) -> bool:
        """
        智能規則基礎的 AI 移動邏輯，使用 A* 路徑規劃，優先收集分數球，避開活著且不可食用鬼魂的動態威脅區域，
        除非在殘局階段、附近有鬼（距離 < 6）或有 3 隻以上有威脅的鬼，否則不主動吃能量球。
        若連續上下交替移動超過 4 次，切換到最近的分數球。
        包含脫困機制以避免卡住。
        
        Args:
            maze (Map): 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
        
        Returns:
            bool: 是否成功設置新目標。
        """
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

        # 步驟 3: 檢查是否進入殘局模式
        is_endgame = len(score_pellets) <= 5

        # 步驟 4: 計算當前移動方向
        direction = None
        if self.last_direction:
            current_dx, current_dy = 0, 0
            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) not in ['#', 'X']:
                    if self.set_new_target(dx, dy, maze):
                        current_dx, current_dy = dx, dy
                        direction = (dx, dy)
                        break
            if direction:
                # 檢測連續上下交替移動
                if abs(current_dy) > 0:
                    if self.last_direction and (self.last_direction[1] * current_dy < 0):
                        self.alternating_vertical_count += 1
                        if self.alternating_vertical_count > 4:
                            if score_pellets:
                                score_options = [(s, (s.x - current_x) ** 2 + (s.y - current_y) ** 2) for s in score_pellets]
                                closest_score = min(score_options, key=lambda x: x[1])[0]
                                goal = (closest_score.x, closest_score.y)
                                direction = self.find_path(start, goal, maze, ghosts, power_pellets, mode="approach", target_type="score")
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
                if not maze.xy_valid(x, y) or maze.get_tile(x, y) in ['#', 'X']:
                    return float('inf')
                for ghost in ghosts:
                    if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                        dist = ((x - ghost.x) ** 2 + (y - ghost.y) ** 2) ** 0.5
                        if dist < 2:
                            return float('inf')
            return ((x - current_x) ** 2 + (y - current_y) ** 2) ** 0.5

        # 步驟 6: 殘局模式處理
        if is_endgame and score_pellets:
            remaining_scores = [(s, (s.x, s.y)) for s in score_pellets]
            current_pos = start
            best_path = []
            while remaining_scores:
                closest = min(remaining_scores, key=lambda s: ((s[1][0] - current_pos[0]) ** 2 + (s[1][1] - current_pos[1]) ** 2) ** 0.5)
                best_path.append(closest[1])
                current_pos = closest[1]
                remaining_scores.remove(closest)

            # 殘局中根據威脅動態決定是否吃能量球
            use_power_pellet = False
            if power_pellets and (min_danger_dist < 6 or alive_ghosts >= 3):
                power_options = [(p, (p.x - current_x) ** 2 + (p.y - current_y) ** 2) for p in power_pellets]
                closest_power = min(power_options, key=lambda x: x[1])[0]
                power_dist = ((closest_power.x - current_x) ** 2 + (closest_power.y - current_y) ** 2) ** 0.5
                if power_dist < sum(((p[0] - current_x) ** 2 + (p[1] - current_y) ** 2) ** 0.5 for p in best_path) / len(best_path):
                    use_power_pellet = True
                    goal = (closest_power.x, closest_power.y)
                    direction = self.find_path(start, goal, maze, ghosts, power_pellets, mode="approach", target_type="power")
                    if direction:
                        dx, dy = direction
                        if self.set_new_target(dx, dy, maze):
                            self.speed = self.boost_speed
                            self.last_direction = (dx, dy)
                            self.stuck_count = 0
                            return True

            if not use_power_pellet and best_path:
                goal = best_path[0]
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, mode="approach", target_type="score")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True

        # 步驟 7: 優先級策略
        if min_danger_dist < 6 or alive_ghosts >= 3:
            self.speed = self.boost_speed
            if power_pellets:
                power_options = [(p, (p.x - current_x) ** 2 + (p.y - current_y) ** 2) for p in power_pellets]
                closest_power = min(power_options, key=lambda x: x[1])[0]
                goal = (closest_power.x, closest_power.y)
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, mode="approach", target_type="power")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True
            else:
                direction = self.find_path(start, None, maze, ghosts, power_pellets, mode="flee", target_type="none")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True
        else:
            # 追逐可食用鬼魂
            edible_ghosts = [ghost for ghost in ghosts if ghost.edible and ghost.edible_timer > 0]
            if edible_ghosts:
                closest_edible = min(edible_ghosts, key=lambda g: (g.x - current_x) ** 2 + (g.y - current_y) ** 2)
                goal = (closest_edible.x, closest_edible.y)
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, mode="approach", target_type="edible")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True
            
            # 收集分數球
            if score_pellets:
                score_options = [(s, (s.x - current_x) ** 2 + (s.y - current_y) ** 2) for s in score_pellets]
                closest_score = min(score_options, key=lambda x: x[1])[0]
                goal = (closest_score.x, closest_score.y)
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, mode="approach", target_type="score")
                if direction:
                    dx, dy = direction
                    if self.set_new_target(dx, dy, maze):
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True
            
            

        # 步驟 8: 選擇最安全方向或脫困
        safe_directions = [d for d in directions if maze.xy_valid(current_x + d[0], current_y + d[1]) 
                         and maze.get_tile(current_x + d[0], current_y + d[1]) not in ['#', 'X']]
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
                    self.speed = self.boost_speed
                    self.last_direction = (dx, dy)
                    self.stuck_count = 0
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
        self.speed = 1.0
        self.edible = False
        self.edible_timer = 0
        self.respawn_timer = 0
        self.returning_to_spawn = False
        self.return_speed = 2.5
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
                self.waiting = False
                self.speed = 2.0
            return
        
        if self.edible:
            self.edible_timer -= 1
            if self.edible_timer <= 0:
                self.edible = False
                self.edible_timer = 0
                # print(f"{self.name} is no longer edible.")
        
        if self.respawn_timer > 0:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0 and self.returning_to_spawn:
                self.reset_position(maze, [(x, y) for x, y in [(x, y) for y in range(maze.h) 
                                                               for x in range(maze.w) if maze.get_tile(x, y) == 'S']])
            return
        
        if self.returning_to_spawn:
            self.return_to_spawn(maze)
        elif self.edible and self.edible_timer > 0:
            self.escape_from_pacman(pacman, maze)
        else:
            self.chase_pacman(pacman, maze)

    def return_to_spawn(self, maze):
        """
        快速返回最近的重生點 'S'.
        
        Args:
            maze (Map): 迷宮物件。
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
        if self.wait_timer > 0 or self.returning_to_spawn or self.respawn_timer > 0 or self.waiting:
            pass
        else:
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