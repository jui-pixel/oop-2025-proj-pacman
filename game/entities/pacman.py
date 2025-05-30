# game/entities/pacman.py
"""
定義 Pac-Man 實體，包括移動邏輯、得分機制和 AI 路徑規劃。
"""
from .entity_base import Entity
from heapq import heappush, heappop
from typing import Tuple, List
import random
from config import *
from .pellets import PowerPellet, ScorePellet
from .ghost import Ghost


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
        self.speed = 4.0  # 基礎移動速度
        self.last_direction = None
        self.alternating_vertical_count = 0
        self.stuck_count = 0
        self.max_stuck_frames = 10

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
        
        Args:
            start (tuple): 起點 (x, y)。
            goal (tuple): 目標 (x, y)，在逃跑模式下可為 None。
            maze: 迷宮物件。
            ghosts (List[Ghost]): 鬼魂列表。
            power_pellets (List[PowerPellet]): 能量球列表。
            mode (str): "approach" 或 "flee"。
            target_type (str): 目標類型 ("score", "power", "edible")。
        
        Returns:
            tuple: 第一步的方向 (dx, dy)，如果無路徑則返回隨機安全方向。
        """
        def heuristic(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        predicted_danger = set()
        for ghost in ghosts:
            if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                dx = 1 if ghost.x < start[0] else -1 if ghost.x > start[0] else 0
                dy = 1 if ghost.y < start[1] else -1 if ghost.y > start[1] else 0
                next_x, next_y = ghost.x + dx, ghost.y + dy
                if maze.xy_valid(next_x, next_y):
                    predicted_danger.add((next_x, next_y))

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
                    return (next_pos[0] - start[0], next_pos[1] - start[1])
                return None

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

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        safe_directions = [(dx, dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                          if maze.xy_valid(start[0] + dx, start[1] + dy)
                          and maze.get_tile(start[0] + dx, start[1] + dy) not in ['#', 'X']]
        return random.choice(safe_directions) if safe_directions else None

    def rule_based_ai_move(self, maze, power_pellets: List['PowerPellet'], score_pellets: List['ScorePellet'], ghosts: List['Ghost']) -> bool:
        """
        智能規則基礎的 AI 移動邏輯。
        
        Args:
            maze: 迷宮物件。
            power_pellets (List[PowerPellet]): 能量球列表。
            score_pellets (List[ScorePellet]): 分數球列表。
            ghosts (List[Ghost]): 鬼魂列表。
        
        Returns:
            bool: 是否成功設置新目標。
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        current_x, current_y = self.x, self.y
        start = (current_x, current_y)

        alive_ghosts = sum(1 for ghost in ghosts if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible)
        min_danger_dist = min(((current_x - ghost.x) ** 2 + (current_y - ghost.y) ** 2) ** 0.5
                             for ghost in ghosts if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible) if alive_ghosts else float('inf')
        is_endgame = len(score_pellets) <= 10

        if self.last_direction:
            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) not in ['#', 'X']:
                    if self.set_new_target(dx, dy, maze):
                        if abs(dy) > 0 and self.last_direction and (self.last_direction[1] * dy < 0):
                            self.alternating_vertical_count += 1
                            if self.alternating_vertical_count > 4 and score_pellets:
                                closest_score = min(score_pellets, key=lambda s: (s.x - current_x) ** 2 + (s.y - current_y) ** 2)
                                goal = (closest_score.x, closest_score.y)
                                direction = self.find_path(start, goal, maze, ghosts, power_pellets, "approach", "score")
                                if direction and self.set_new_target(direction[0], direction[1], maze):
                                    self.last_direction = direction
                                    self.alternating_vertical_count = 0
                                    self.stuck_count = 0
                                    return True
                        else:
                            self.alternating_vertical_count = 0
                        self.last_direction = (dx, dy)
                        self.stuck_count = 0
                        return True

        if is_endgame and score_pellets:
            remaining_scores = [(s, (s.x, s.y)) for s in score_pellets]
            current_pos = start
            best_path = []
            while remaining_scores:
                closest = min(remaining_scores, key=lambda s: ((s[1][0] - current_pos[0]) ** 2 + (s[1][1] - current_pos[1]) ** 2) ** 0.5)
                best_path.append(closest[1])
                current_pos = closest[1]
                remaining_scores.remove(closest)
            if best_path and (min_danger_dist < 6 or alive_ghosts >= 3) and power_pellets:
                closest_power = min(power_pellets, key=lambda p: (p.x - current_x) ** 2 + (p.y - current_y) ** 2)
                if ((closest_power.x - current_x) ** 2 + (closest_power.y - current_y) ** 2) ** 0.5 < sum(((p[0] - current_x) ** 2 + (p[1] - current_y) ** 2) ** 0.5 for p in best_path) / len(best_path):
                    goal = (closest_power.x, closest_power.y)
                    direction = self.find_path(start, goal, maze, ghosts, power_pellets, "approach", "power")
                    if direction and self.set_new_target(direction[0], direction[1], maze):
                        self.last_direction = direction
                        self.stuck_count = 0
                        return True
            if best_path:
                goal = best_path[0]
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, "approach", "score")
                if direction and self.set_new_target(direction[0], direction[1], maze):
                    self.last_direction = direction
                    self.stuck_count = 0
                    return True

        if min_danger_dist < 6 or alive_ghosts >= 3:
            if power_pellets:
                closest_power = min(power_pellets, key=lambda p: (p.x - current_x) ** 2 + (p.y - current_y) ** 2)
                goal = (closest_power.x, closest_power.y)
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, "approach", "power")
                if direction and self.set_new_target(direction[0], direction[1], maze):
                    self.last_direction = direction
                    self.stuck_count = 0
                    return True
                direction = self.find_path(start, None, maze, ghosts, power_pellets, "flee", "none")
                if direction and self.set_new_target(direction[0], direction[1], maze):
                    self.last_direction = direction
                    self.stuck_count = 0
                    return True
        else:
            edible_ghosts = [ghost for ghost in ghosts if ghost.edible and ghost.edible_timer > 0]
            if edible_ghosts:
                closest_edible = min(edible_ghosts, key=lambda g: (g.x - current_x) ** 2 + (g.y - current_y) ** 2)
                if ((closest_edible.x - current_x) ** 2 + (closest_edible.y - current_y) ** 2) < 15:
                    goal = (closest_edible.x, closest_edible.y)
                    direction = self.find_path(start, goal, maze, ghosts, power_pellets, "approach", "edible")
                    if direction and self.set_new_target(direction[0], direction[1], maze):
                        self.last_direction = direction
                        self.stuck_count = 0
                        return True
            if score_pellets:
                closest_score = min(score_pellets, key=lambda s: (s.x - current_x) ** 2 + (s.y - current_y) ** 2)
                goal = (closest_score.x, closest_score.y)
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, "approach", "score")
                if direction and self.set_new_target(direction[0], direction[1], maze):
                    self.last_direction = direction
                    self.stuck_count = 0
                    return True
            elif edible_ghosts:
                closest_edible = min(edible_ghosts, key=lambda g: (g.x - current_x) ** 2 + (g.y - current_y) ** 2)
                goal = (closest_edible.x, closest_edible.y)
                direction = self.find_path(start, goal, maze, ghosts, power_pellets, "approach", "edible")
                if direction and self.set_new_target(direction[0], direction[1], maze):
                    self.last_direction = direction
                    self.stuck_count = 0
                    return True

        safe_directions = [d for d in directions if maze.xy_valid(current_x + d[0], current_y + d[1])
                         and maze.get_tile(current_x + d[0], current_y + d[1]) not in ['#', 'X']]
        if not safe_directions:
            return False

        best_direction = min(safe_directions, key=lambda d: ((current_x + d[0] - current_x) ** 2 + (current_y + d[1] - current_y) ** 2) ** 0.5)
        if self.set_new_target(best_direction[0], best_direction[1], maze):
            self.last_direction = best_direction
            self.stuck_count = 0
            return True
        self.stuck_count += 1
        if self.stuck_count > self.max_stuck_frames:
            random_direction = random.choice(safe_directions)
            if self.set_new_target(random_direction[0], random_direction[1], maze):
                self.last_direction = random_direction
                self.stuck_count = 0
                return True
        return False