# game/entities/pacman.py
"""
定義 Pac-Man 類，實現其行為邏輯，包括移動、吃球和行為樹 AI。

這個類負責處理 Pac-Man 的移動決策、與道具和鬼魂的交互，以及基於行為樹的智能行為。
"""

# 匯入必要的模組
from enum import Enum  # 用於定義行為樹節點狀態
from typing import Tuple, List, Optional  # 用於型別提示
from .entity_base import Entity  # 匯入實體基類
from heapq import heappush, heappop  # 用於 A* 算法的優先級隊列
import random  # 用於隨機選擇方向
# 從 config 檔案匯入常數
from config import CELL_SIZE, TILE_BOUNDARY, TILE_WALL, TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR, PACMAN_BASE_SPEED, PACMAN_AI_SPEED, MAX_STUCK_FRAMES
# 匯入相關實體類
from .pellets import PowerPellet, ScorePellet
from .ghost import Ghost

# 定義行為樹節點狀態的枚舉類
class NodeStatus(Enum):
    SUCCESS = "success"  # 節點執行成功
    FAILURE = "failure"  # 節點執行失敗
    RUNNING = "running"  # 節點正在執行

# 定義行為樹基類
class BehaviorNode:
    def execute(self, pacman: 'PacMan', maze, power_pellets: List[PowerPellet], score_pellets: List[ScorePellet], ghosts: List[Ghost]) -> NodeStatus:
        """
        執行行為樹節點的邏輯。

        這個方法由子類實現，定義具體的行為邏輯。
        """
        raise NotImplementedError

# 定義條件節點類
class ConditionNode(BehaviorNode):
    def __init__(self, condition_func):
        """
        初始化條件節點，設置條件函數。

        Args:
            condition_func: 條件檢查函數，返回布林值。
        """
        self.condition_func = condition_func

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        """
        執行條件檢查，返回成功或失敗狀態。

        Returns:
            NodeStatus: SUCCESS（條件為真）或 FAILURE（條件為假）。
        """
        return NodeStatus.SUCCESS if self.condition_func(pacman, maze, power_pellets, score_pellets, ghosts) else NodeStatus.FAILURE

# 定義動作節點類
class ActionNode(BehaviorNode):
    def __init__(self, action_func):
        """
        初始化動作節點，設置動作函數。

        Args:
            action_func: 動作執行函數，返回 NodeStatus。
        """
        self.action_func = action_func

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        """
        執行動作，返回動作結果狀態。

        Returns:
            NodeStatus: 動作函數的執行結果。
        """
        return self.action_func(pacman, maze, power_pellets, score_pellets, ghosts)

# 定義選擇節點類
class SelectorNode(BehaviorNode):
    def __init__(self, children: List[BehaviorNode]):
        """
        初始化選擇節點，設置子節點列表。

        Args:
            children: 子節點列表。
        """
        self.children = children

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        """
        依次執行子節點，直到某個子節點成功或正在運行。

        原理：
        - 選擇節點會按順序嘗試每個子節點。
        - 若某子節點返回 SUCCESS 或 RUNNING，則立即返回該狀態。
        - 若所有子節點都失敗，則返回 FAILURE。

        Returns:
            NodeStatus: 子節點的執行結果。
        """
        for child in self.children:
            status = child.execute(pacman, maze, power_pellets, score_pellets, ghosts)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        return NodeStatus.FAILURE

# 定義序列節點類
class SequenceNode(BehaviorNode):
    def __init__(self, children: List[BehaviorNode]):
        """
        初始化序列節點，設置子節點列表。

        Args:
            children: 子節點列表。
        """
        self.children = children

    def execute(self, pacman, maze, power_pellets, score_pellets, ghosts):
        """
        依次執行子節點，直到某個子節點失敗或正在運行。

        原理：
        - 序列節點按順序執行所有子節點。
        - 若某子節點返回 FAILURE 或 RUNNING，則立即返回該狀態。
        - 若所有子節點都成功，則返回 SUCCESS。

        Returns:
            NodeStatus: 子節點的執行結果。
        """
        for child in self.children:
            status = child.execute(pacman, maze, power_pellets, score_pellets, ghosts)
            if status == NodeStatus.FAILURE:
                return NodeStatus.FAILURE
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        return NodeStatus.SUCCESS

class PacMan(Entity):
    def __init__(self, x: int, y: int):
        """
        初始化 Pac-Man，設置初始屬性和行為樹。

        Args:
            x (int): 初始 x 坐標（格子坐標）。
            y (int): 初始 y 坐標（格子坐標）。
        """
        # 調用基類 Entity 初始化，設置符號 'P'
        super().__init__(x, y, 'P')
        # 初始化分數
        self.score = 0
        # 初始化生命數
        self.lives = 3
        # 初始化存活狀態
        self.alive = True
        # 設置基礎移動速度
        self.speed = PACMAN_BASE_SPEED
        # 初始化上次移動方向
        self.last_direction = None
        # 初始化垂直交替移動計數器
        self.alternating_vertical_count = 0
        # 初始化卡住計數器
        self.stuck_count = 0
        # 設置最大卡住幀數
        self.max_stuck_frames = MAX_STUCK_FRAMES
        # 記錄初始位置
        self.initial_x = x
        self.initial_y = y
        # 構建行為樹
        self.behavior_tree = self._build_behavior_tree()

    def _build_behavior_tree(self) -> BehaviorNode:
        """
        構建行為樹，模擬基於規則的 AI 移動邏輯。

        原理：
        - 行為樹由條件節點和動作節點組成，按照優先順序組織決策邏輯。
        - 包括以下策略：
          1. 如果有即時威脅（鬼魂距離 ≤ 1），則逃跑。
          2. 如果附近有威脅（鬼魂距離 < 6）且有能量球，則追尋能量球。
          3. 如果是末尾階段（剩餘分數球 ≤ 10），優先選擇較近的能量球或分數球。
          4. 如果有可食用鬼魂（距離 < 10 且可食用時間 > 3），則追逐鬼魂。
          5. 否則追尋分數球。
          6. 如果卡住，執行隨機安全移動。

        Returns:
            BehaviorNode: 構建完成的行為樹根節點。
        """
        # 定義條件函數
        def is_immediate_threat(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查是否有即時威脅（鬼魂距離 ≤ 1）。
            """
            min_danger_dist = float('inf')
            for ghost in ghosts:
                if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                    dist = ((pacman.x - ghost.x) ** 2 + (pacman.y - ghost.y) ** 2) ** 0.5
                    min_danger_dist = min(min_danger_dist, dist)
            return min_danger_dist <= 1

        def is_threat_nearby(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查是否有附近威脅（鬼魂距離 < 6）。
            """
            min_danger_dist = float('inf')
            for ghost in ghosts:
                if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                    dist = ((pacman.x - ghost.x) ** 2 + (pacman.y - ghost.y) ** 2) ** 0.5
                    min_danger_dist = min(min_danger_dist, dist)
            return min_danger_dist < 6

        def has_power_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查是否還有能量球。
            """
            return bool(power_pellets)

        def is_endgame(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查是否進入末遊戲階段（剩餘分數球 ≤ 10）。
            """
            return len(score_pellets) <= 10

        def is_power_pellet_closer(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查能量球是否比分數球平均距離更近。
            """
            if not power_pellets or not score_pellets:
                return False
            power_dist = min(((p.x - pacman.x) ** 2 + (p.y - pacman.y) ** 2) ** 0.5 for p in power_pellets)
            score_dists = [((s.x - pacman.x) ** 2 + (s.y - pacman.y) ** 2) ** 0.5 for s in score_pellets]
            avg_score_dist = sum(score_dists) / len(score_dists)
            return power_dist < avg_score_dist

        def has_edible_ghost(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查是否有可食用的鬼魂（距離 < 10 且可食用時間 > 3）。
            """
            for ghost in ghosts:
                if ghost.edible and ghost.edible_timer > 3 and not ghost.returning_to_spawn and not ghost.waiting:
                    dist = ((ghost.x - pacman.x) ** 2 + (ghost.y - pacman.y) ** 2) ** 0.5
                    if dist < 10:
                        return True
            return False

        def has_score_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查是否還有分數球。
            """
            return bool(score_pellets)

        def is_stuck(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            檢查 Pac-Man 是否卡住（卡住計數 > 最大限度）。
            """
            return pacman.stuck_count > pacman.max_stuck_frames

        # 定義動作函數
        def flee_from_ghosts(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            逃離鬼魂，選擇最安全的方向。
            """
            direction = pacman.find_path((pacman.x, pacman.y), None, maze, ghosts, score_pellets, power_pellets, mode="flee", target_type="none")
            if direction:
                dx, dy = direction
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def move_to_power_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            移動到最近的能量球。
            """
            closest_power = min(power_pellets, key=lambda p: (p.x - pacman.x) ** 2 + (p.y - pacman.y) ** 2)
            goal = (closest_power.x, closest_power.y)
            direction = pacman.find_path((pacman.x, pacman.y), goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="power")
            if direction:
                dx, dy = direction
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    # print(pacman.last_direction)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def move_to_score_pellet(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            移動到最近的分數球。
            """
            if not score_pellets:
                return NodeStatus.FAILURE
            closest_score = min(score_pellets, key=lambda s: (s.x - pacman.x) ** 2 + (s.y - pacman.y) ** 2)
            goal = (closest_score.x, closest_score.y)
            direction = pacman.find_path((pacman.x, pacman.y), goal, maze, ghosts, score_pellets, power_pellets, mode="approach", target_type="score")
            if direction:
                dx, dy = direction
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    # print(pacman.last_direction)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def move_to_edible_ghost(pacman: PacMan, maze, power_pellets, score_pellets, ghosts: List[Ghost]):
            """
            移動到最近的可食用鬼魂。
            """
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
                    # print(pacman.last_direction)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        def random_safe_move(pacman, maze, power_pellets, score_pellets, ghosts):
            """
            執行隨機安全移動。
            """
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            safe_directions = [(dx, dy) for dx, dy in directions 
                     if maze.xy_valid(pacman.x + dx, pacman.y + dy) 
                     and maze.get_tile(pacman.x + dx, pacman.y + dy) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
            if safe_directions:
                dx, dy = random.choice(safe_directions)
                if pacman.set_new_target(dx, dy, maze):
                    pacman.last_direction = (dx, dy)
                    # print(pacman.last_direction)
                    pacman.stuck_count = 0
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        # 構建行為樹
        return SelectorNode([
            # 即時威脅：逃跑
            SequenceNode([
                ConditionNode(is_immediate_threat),
                ActionNode(flee_from_ghosts)
            ]),
            # 附近威脅且有能量球：追尋能量球
            SequenceNode([
                ConditionNode(is_threat_nearby),
                ConditionNode(has_power_pellet),
                ActionNode(move_to_power_pellet)
            ]),
            # 末遊戲階段：優先能量球或分數球
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
            # 有可食用鬼魂：追逐鬼魂
            SequenceNode([
                ConditionNode(has_edible_ghost),
                ActionNode(move_to_edible_ghost)
            ]),
            # 有分數球：追尋分數球
            SequenceNode([
                ConditionNode(has_score_pellet),
                ActionNode(move_to_score_pellet)
            ]),
            # 卡住：隨機移動
            SequenceNode([
                ConditionNode(is_stuck),
                ActionNode(random_safe_move)
            ])
        ])

    def rule_based_ai_move(self, maze, power_pellets: List['PowerPellet'], score_pellets: List['ScorePellet'], ghosts: List['Ghost']) -> bool:
        """
        使用行為樹執行 AI 移動邏輯，替代原有的多層決策。

        Args:
            maze: 迷宮物件，提供路徑信息。
            power_pellets: 能量球列表。
            score_pellets: 分數球列表。
            ghosts: 鬼魂列表。

        Returns:
            bool: 是否成功設置新目標。
        """
        # 設置 AI 移動速度
        self.speed = PACMAN_AI_SPEED
        # 執行行為樹
        status = self.behavior_tree.execute(self, maze, power_pellets, score_pellets, ghosts)
        # 返回是否成功設置新目標
        return status == NodeStatus.SUCCESS

    def eat_pellet(self, pellets: List['PowerPellet']) -> int:
        """
        檢查並吃掉能量球，更新分數並移除能量球。

        原理：
        - 若 Pac-Man 位置與能量球重合，增加分數並移除該能量球。
        - 分數增量等於能量球的 value（預設為 10）。

        Args:
            plates (List[PowerPellet]): 能量球列表。

        Returns:
            int: 增加的分數（若未吃到則為 0）。
        """
        for pellet in pellets[:]:
            # 檢查是否與能量球位置重合
            if pellet.x == self.x and pellet.y == self.y:
                # 增加分數
                self.score += pellet.value
                # 移除能量球
                pellets.remove(pellet)
                # 返回分數增量
                return pellet.value
        return 0

    def eat_score_pellet(self, score_pellets: List['ScorePellet']) -> int:
        """
        檢查並吃掉分數球，更新分數並移除分數球。

        原理：
        - 若 Pac-Man 位置與分數球重合，增加分數並移除該分數球。
        - 分數增量等於分數球的 value（預設為 2）。

        Args:
            score_pellets (List[ScorePellet]): 分數球列表。

        Returns:
            int: 增加的分數（若未吃到則為 0）。
        """
        for score_pellet in score_pellets[:]:
            # 檢查是否與分數球位置重合
            if score_pellet.x == self.x and score_pellet.y == self.y:
                # 增加分數
                self.score += score_pellet.value
                # 移除分數球
                score_pellets.remove(score_pellet)
                # 返回分數增量
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
        # 扣除一條命
        self.lives -= 1
        # 扣除 50 分
        self.score -= 50
        # # 重置位置到初始坐標
        # self.x = self.initial_x
        # self.y = self.initial_y
        # # 重置像素坐標
        # self.current_x = self.x * CELL_SIZE + CELL_SIZE // 2
        # self.current_y = self.y * CELL_SIZE + CELL_SIZE // 2
        # # 重置目標坐標
        # self.target_x = self.x
        # self.target_y = self.y
        # # 清空上次移動方向
        # self.last_direction = None
        # # 清空垂直交替計數
        # self.alternating_vertical_count = 0
        # # 清空卡住計數
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
            """
            計算啟發式函數（歐幾里得距離）。
            """
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        # 預測鬼魂下一個位置
        predicted_danger = set()
        for ghost in ghosts:
            if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                # 根據鬼魂與 Pac-Man 的相對位置預測移動方向
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
            # 搜尋範圍為迷宮尺寸的一半
            search_radius = min(maze.width, maze.height) // 2
            # 遍歷搜尋範圍內的格子
            for y in range(max(0, center_y - search_radius), min(maze.height, center_y + search_radius + 1)):
                for x in range(max(0, center_x - search_radius), min(maze.width, center_x + search_radius + 1)):
                    # 檢查是否為安全格子
                    if maze.get_tile(x, y) not in [TILE_BOUNDARY, TILE_WALL] and (x, y) not in predicted_danger:
                        # 計算到最近危險鬼魂的距離
                        min_ghost_dist = min(((x - ghost.x) ** 2 + (y - ghost.y) ** 2) ** 0.5 for ghost in danger_ghosts)
                        if min_ghost_dist > max_dist:
                            max_dist = min_ghost_dist
                            best_goal = (x, y)
            goal = best_goal if best_goal else start

        # 初始化 A* 算法的數據結構
        open_set = []
        heappush(open_set, (0, start))  # 優先級隊列，存儲 (f_score, 節點)
        came_from = {}  # 記錄路徑來源
        g_score = {start: 0}  # 記錄到起點的實際成本
        f_score = {start: heuristic(start, goal)}  # 記錄估計總成本
        closed_set = set()  # 已訪問節點集合

        # 定義威脅區域
        danger_obstacles = set()
        for ghost in ghosts:
            if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                # 根據與鬼魂的距離計算威脅半徑
                threat_radius = 1 + int(min(2, max(0, 6 - ((start[0] - ghost.x) ** 2 + (start[1] - ghost.y) ** 2) ** 0.5)))
                for dx in range(-threat_radius, threat_radius + 1):
                    for dy in range(-threat_radius, threat_radius + 1):
                        x, y = ghost.x + dx, ghost.y + dy
                        if maze.xy_valid(x, y):
                            danger_obstacles.add((x, y))

        # A* 算法主循環
        while open_set:
            _, current = heappop(open_set)  # 取出 f_score 最小的節點

            # 如果到達目標，返回路徑的第一步
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

            # 檢查四個方向的鄰居
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                # 檢查鄰居是否有效且可通行
                if not maze.xy_valid(neighbor[0], neighbor[1]) or maze.get_tile(neighbor[0], neighbor[1]) in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]:
                    continue

                # 計算鬼魂威脅成本
                danger_cost = 0
                for ghost in ghosts:
                    if not ghost.returning_to_spawn and not ghost.waiting and not ghost.edible:
                        dist = ((neighbor[0] - ghost.x) ** 2 + (neighbor[1] - ghost.y) ** 2) ** 0.5
                        if dist < 4:
                            danger_cost += 1500 / max(1, dist)  # 靠近鬼魂加重成本
                        elif neighbor in predicted_danger:
                            danger_cost += 2000  # 預測鬼魂位置加重成本
                
                # 計算能量球避免懲罰（追逐可食用鬼魂時）
                power_avoidance_penalty = 0
                if target_type == "edible" and power_pellets:
                    for pellet in power_pellets:
                        dist = ((neighbor[0] - pellet.x) ** 2 + (neighbor[1] - pellet.y) ** 2) ** 0.5
                        if dist < 2:
                            power_avoidance_penalty += 100 / max(1, dist)

                # 計算能量球懲罰（追逐分數球時）
                power_penalty = 0
                if target_type == "score" and power_pellets:
                    for pellet in power_pellets:
                        dist = (neighbor[0] == pellet.x) and (neighbor[1] == pellet.y)
                        if dist:
                            power_penalty += 1000

                # 計算分數球獎勵
                score_reward = 0
                if target_type in ["edible", "none", "flee"] and maze.get_tile(neighbor[0], neighbor[1]) != TILE_WALL:
                    for pellet in score_pellets:
                        if pellet.x == neighbor[0] and pellet.y == neighbor[1]:
                            score_reward -= 0.9  # 順路吃分數球

                # 計算試探性 g 分數
                tentative_g_score = g_score[current] + 1 + danger_cost + power_avoidance_penalty + power_penalty + score_reward

                # 如果鄰居已訪問且成本更高，則跳過
                if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                # 如果找到更低成本的路徑，更新數據
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