# game/entities/pacman.py
from enum import Enum
from typing import Tuple, List, Optional
from .entity_base import Entity
from heapq import heappush, heappop
import random
from config import CELL_SIZE, TILE_BOUNDARY, TILE_WALL, TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR, PACMAN_BASE_SPEED, PACMAN_AI_SPEED, MAX_STUCK_FRAMES
from .pellets import PowerPellet, ScorePellet
from .ghost import Ghost

# 行為樹節點狀態
class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"

# 行為樹基類
class BehaviorNode:
    def execute(self, pacman: 'PacMan', maze, power_pellets: List[PowerPellet], score_pellets: List[ScorePellet], ghosts: List[Ghost]) -> NodeStatus:
        raise NotImplementedError

# 條件節點
class ConditionNode(BehaviorNode):
    def __init__(self, condition_func):
        self.condition_func = condition_func

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        return NodeStatus.SUCCESS if self.condition_func(pacman, maze, power_pellets, score_pellets, ghosts) else NodeStatus.FAILURE

# 動作節點
class ActionNode(BehaviorNode):
    def __init__(self, action_func):
        self.action_func = action_func

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        return self.action_func(pacman, maze, power_pellets, score_pellets, ghosts)

# 選擇節點
class SelectorNode(BehaviorNode):
    def __init__(self, children: List[BehaviorNode]):
        self.children = children

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        for child in self.children:
            status = child.execute(pacman, maze, power_pellets, score_pellets, ghosts)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        return NodeStatus.FAILURE

# 序列節點
class SequenceNode(BehaviorNode):
    def __init__(self, children: List[BehaviorNode]):
        self.children = children

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        for child in self.children:
            status = child.execute(pacman, maze, power_pellets, score_pellets, ghosts)
            if status == NodeStatus.FAILURE:
                return NodeStatus.FAILURE
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        return NodeStatus.SUCCESS

class PacMan(Entity):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, 'P')
        self.score = 0
        self.lives = 3
        self.alive = True
        self.speed = PACMAN_BASE_SPEED
        self.last_direction = None
        self.alternating_vertical_count = 0
        self.stuck_count = 0
        self.max_stuck_frames = MAX_STUCK_FRAMES
        self.initial_x = x
        self.initial_y = y
        # 初始化行為樹
        self.behavior_tree = self._build_behavior_tree()

    def _build_behavior_tree(self) -> BehaviorNode:
        """構建行為樹，模擬 rule_based_ai_move 的決策邏輯。"""
        # 條件函數
        def is_immediate_threat(pacman, maze, power_pellets, score_pellets, ghosts):
            min_danger_dist = float('inf')
            for ghost in ghosts:
                if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                    dist = ((pacman.x - ghost.x) ** 2 + (pacman.y - ghost.y) ** 2) ** 0.5
                    min_danger_dist = min(min_danger_dist, dist)
            return min_danger_dist <= 1

        def is_threat_nearby(pacman, maze, power_pellets, score_pellets, ghosts):
            min_danger_dist = float('inf')
            for ghost in ghosts:
                if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                    dist = ((pacman.x - ghost.x) ** 2 + (pacman.y - ghost.y) ** 2) ** 0.5
                    min_danger_dist = min(min_danger_dist, dist)
            return min_danger_dist < 6

        def has_power_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            return bool(power_pellets)

        def is_endgame(pacman, maze, power_pellets, score_pellets, ghosts):
            return len(score_pellets) <= 10

        def is_power_pellet_closer(pacman, maze, power_pellets, score_pellets, ghosts):
            if not power_pellets or not score_pellets:
                return False
            power_dist = min(((p.x - pacman.x) ** 2 + (p.y - pacman.y) ** 2) ** 0.5 for p in power_pellets)
            score_dists = [((s.x - pacman.x) ** 2 + (s.y - pacman.y) ** 2) ** 0.5 for s in score_pellets]
            avg_score_dist = sum(score_dists) / len(score_dists)
            return power_dist < avg_score_dist

        def has_edible_ghost(pacman, maze, power_pellets, score_pellets, ghosts):
            for ghost in ghosts:
                if ghost.edible and ghost.edible_timer > 3 and not ghost.returning_to_spawn and not ghost.waiting:
                    dist = ((ghost.x - pacman.x) ** 2 + (ghost.y - pacman.y) ** 2) ** 0.5
                    if dist < 10:
                        return True
            return False

        def has_score_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            return bool(score_pellets)

        def is_stuck(pacman, maze, power_pellets, score_pellets, ghosts):
            return pacman.stuck_count > pacman.max_stuck_frames

        # 動作函數
        def flee_from_ghosts(pacman, maze, power_pellets, score_pellets, ghosts):
            direction = pacman.find_path((pacman.x, pacman.y), None, maze, ghosts, score_pellets, power_pellets, mode="flee", target_type="none")
            if direction:
                dx, dy = direction
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def move_to_power_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            closest_power = min(power_pellets, key=lambda p: (p.x - pacman.x) ** 2 + (p.y - pacman.y) ** 2)
            goal = (closest_power.x, closest_power.y)
            direction = pacman.find_path((pacman.x, pacman.y), goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="power")
            if direction:
                dx, dy = direction
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def move_to_score_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            closest_score = min(score_pellets, key=lambda s: (s.x - pacman.x) ** 2 + (s.y - pacman.y) ** 2)
            goal = (closest_score.x, closest_score.y)
            direction = pacman.find_path((pacman.x, pacman.y), goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score")
            if direction:
                dx, dy = direction
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def move_to_edible_ghost(pacman : PacMan, maze, power_pellets, score_pellets, ghosts : List[Ghost]):
            edible_ghosts = [g for g in ghosts if g.edible and g.edible_timer > 3 and not g.returning_to_spawn and not g.waiting]
            if not edible_ghosts:
                return NodeStatus.FAILURE
            closest_edible = min(edible_ghosts, key=lambda g: (g.x - pacman.x) ** 2 + (g.y - pacman.y) ** 2)
            goal = (closest_edible.target_x, closest_edible.target_y)
            direction = pacman.find_path((pacman.x, pacman.y), goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="edible")
            if direction:
                dx, dy = direction
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def random_safe_move(pacman, maze, power_pellets, score_pellets, ghosts):
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            safe_directions = [(dx, dy) for dx, dy in directions 
                              if maze.xy_valid(pacman.x + dx, pacman.y + dy) 
                              and maze.get_tile(pacman.x + dx, pacman.y + dy) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
            if safe_directions:
                dx, dy = random.choice(safe_directions)
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        # 構建行為樹
        return SelectorNode([
            SequenceNode([
                ConditionNode(is_immediate_threat),
                ActionNode(flee_from_ghosts)
            ]),
            SequenceNode([
                ConditionNode(is_threat_nearby),
                ConditionNode(has_power_pellet),
                ActionNode(move_to_power_pellet)
            ]),
            SequenceNode([
                ConditionNode(is_endgame),
                SelectorNode([
                    SequenceNode([
                        ConditionNode(is_power_pellet_closer),
                        ActionNode(move_to_power_pellet)
                    ]),
                    ActionNode(move_to_score_pellet)
                ])
            ]),
            SequenceNode([
                ConditionNode(has_edible_ghost),
                ActionNode(move_to_edible_ghost)
            ]),
            SequenceNode([
                ConditionNode(has_score_pellet),
                ActionNode(move_to_score_pellet)
            ]),
            SequenceNode([
                ConditionNode(is_stuck),
                ActionNode(random_safe_move)
            ])
        ])

    def rule_based_ai_move(self, maze, power_pellets: List['PowerPellet'], score_pellets: List['ScorePellet'], ghosts: List['Ghost']) -> bool:
        """
        使用行為樹執行 AI 移動邏輯，替代原有的多層決策。

        Args:
            maze: 迷宮物件。
            power_pellets: 能量球列表。
            score_pellets: 分數球列表。
            ghosts: 鬼魂列表。

        Returns:
            bool: 是否成功設置新目標。
        """
        self.speed = PACMAN_AI_SPEED
        status = self.behavior_tree.execute(self, maze, power_pellets, score_pellets, ghosts)
        return status == NodeStatus.SUCCESS

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
                        if dist < 4:
                            danger_cost += 1500 / max(1, dist)  # 靠近鬼魂加重成本
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
    