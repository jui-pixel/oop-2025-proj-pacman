# game/entities/ghost.py
"""
定義基礎鬼魂類別（Ghost）及其子類（Ghost1, Ghost2, Ghost3, Ghost4），提供通用行為和特定的追逐策略。

這個模組實現了 Pac-Man 遊戲中的敵人鬼魂，每個鬼魂有獨特的行為策略，增加遊戲挑戰性。
"""

# 匯入必要的模組
from .entity_base import Entity  # 匯入實體基類，提供基本的坐標和移動功能
from ..maze_generator import Map  # 匯入迷宮類，用於檢查格子類型和有效性
from typing import Tuple, List, Optional  # 用於型別提示，提升程式碼可讀性
from collections import deque  # 用於 BFS（廣度優先搜尋）算法的隊列
import random  # 用於隨機選擇方向，增加鬼魂行為的多樣性
# 從 config 檔案匯入常數，例如顏色、瓦片類型和速度設置
from config import RED, PINK, CYAN, LIGHT_BLUE, TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN, GHOST_DEFAULT_SPEED, GHOST_RETURN_SPEED, GHOST_WAIT_TIME, GHOST1_SPEED, GHOST2_SPEED, GHOST3_SPEED, GHOST4_SPEED, PACMAN_AI_SPEED

class Ghost(Entity):
    def __init__(self, x: int, y: int, name: str = "Ghost", color: Tuple[int, int, int] = RED):
        """
        初始化基礎鬼魂，設置位置、名稱、顏色和狀態屬性。

        原理：
        - 鬼魂是 Pac-Man 遊戲中的主要敵人，具有不同狀態（正常、可食用、返回重生點、等待）。
        - 每個鬼魂有獨特的顏色和名稱，支持多樣化的行為策略。
        - 狀態屬性包括：
          - edible：是否可食用（吃能量球後觸發）。
          - returning_to_spawn：是否正在返回重生點（被吃後觸發）。
          - waiting：是否在重生點等待。
          - death_count：死亡次數，影響速度和等待時間。

        Args:
            x (int): 迷宮中的 x 坐標（格子坐標）。
            y (int): 迷宮中的 y 坐標（格子坐標）。
            name (str): 鬼魂名稱，預設為 "Ghost"。
            color (Tuple[int, int, int]): 鬼魂的 RGB 顏色，預設為紅色。
        """
        # 調用基類 Entity 的初始化方法，設置坐標和符號 'G'
        super().__init__(x, y, 'G')
        # 設置鬼魂名稱，用於區分不同鬼魂
        self.name = name
        # 設置鬼魂顏色，用於遊戲渲染
        self.color = color
        # 設置默認移動速度（像素/幀）
        self.default_speed = GHOST_DEFAULT_SPEED
        # 設置當前速度，初始等於默認速度
        self.speed = self.default_speed
        # 初始化是否可食用狀態（吃能量球後為 True）
        self.edible = False
        # 初始化可食用計時器（記錄剩餘可食用幀數）
        self.edible_timer = 0
        # 初始化是否返回重生點狀態（被吃後為 True）
        self.returning_to_spawn = False
        # 設置返回重生點速度（通常較快）
        self.return_speed = GHOST_RETURN_SPEED
        # 初始化死亡次數（影響速度和等待時間）
        self.death_count = 0
        # 初始化是否在等待狀態（到達重生點後為 True）
        self.waiting = False
        # 初始化等待計時器（記錄等待剩餘幀數）
        self.wait_timer = 0
        # 設置初始透明度（用於渲染效果，255 表示完全不透明）
        self.alpha = 255
        # 初始化上次位置（避免反覆移動）
        self.last_x = None
        self.last_y = None
        # 初始化記憶的 Pac-Man 位置（用於追逐策略）
        self.memory_x = x
        self.memory_y = y

    def move(self, pacman, maze, fps: int, ghosts: List['Ghost'] = None):
        """
        根據鬼魂狀態執行移動邏輯（等待、可食用、返回重生點或追逐）。

        原理：
        - 根據當前狀態執行不同移動邏輯：
          - 等待狀態：減少等待計時器，到達 0 時恢復正常狀態並加速。
          - 可食用狀態：執行逃跑邏輯，遠離 Pac-Man。
          - 返回重生點：快速返回迷宮中的 'S' 格子。
          - 正常狀態：執行子類定義的追逐策略。
        - 速度隨死亡次數增加：new_speed = default_speed * (1.1 ^ death_count)。

        Args:
            pacman: PacMan 物件，提供位置信息。
            maze: 迷宮物件，提供路徑信息。
            fps (int): 每秒幀數，用於計算移動距離。
            ghosts (List[Ghost], optional): 其他鬼魂列表，用於協作。
        """
        # 如果鬼魂在等待狀態
        if self.waiting:
            # 減少等待計時器
            self.wait_timer -= 1
            # 當計時器到達 0 時，結束等待狀態
            if self.wait_timer <= 0:
                self.waiting = False
                # 根據死亡次數調整速度，速度隨死亡次數增加，但不超過 Pac-Man 的 AI 速度
                self.speed = min(self.default_speed * (1.1 ** self.death_count), PACMAN_AI_SPEED)
            return

        # 如果鬼魂在可食用狀態
        if self.edible:
            # 減少可食用計時器
            self.edible_timer -= 1
            # 當計時器到達 0 時，結束可食用狀態
            if self.edible_timer <= 0:
                self.edible = False
                self.edible_timer = 0

        # 根據狀態執行移動
        if self.returning_to_spawn:
            # 返回重生點
            self.return_to_spawn(maze)
        elif self.edible and self.edible_timer > 0:
            # 逃離 Pac-Man
            self.escape_from_pacman(pacman, maze)
        else:
            # 正常狀態下追逐 Pac-Man
            self.chase_pacman(pacman, maze, ghosts)

    def bfs_path(self, start_x: int, start_y: int, target_x: int, target_y: int, maze) -> Optional[Tuple[int, int]]:
        """
        使用廣度優先搜尋（BFS）尋找從 (start_x, start_y) 到 (target_x, target_y) 的最短路徑。

        原理：
        - BFS 是一種圖搜尋算法，保證找到最短路徑，適用於迷宮中的路徑規劃。
        - 檢查可通行格子（TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN）。
        - 若無直接路徑，嘗試最近的可通行點作為替代目標。
        - 返回第一步的方向 (dx, dy)，或 None 表示無路徑。

        Args:
            start_x (int): 起始 x 坐標。
            start_y (int): 起始 y 坐標。
            target_x (int): 目標 x 坐標。
            target_y (int): 目標 y 坐標。
            maze: 迷宮物件，提供瓦片信息。

        Returns:
            Optional[Tuple[int, int]]: 第一步方向 (dx, dy)，或 None。
        """
        # 初始化 BFS 隊列，儲存 (x, y, 路徑)
        queue = deque([(start_x, start_y, [])])
        # 初始化已訪問節點集合，避免重複訪問
        visited = {(start_x, start_y)}
        # 定義四個移動方向：下、上、右、左
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # BFS 主循環
        while queue:
            # 取出隊列頭部節點
            x, y, path = queue.popleft()
            # 如果到達目標，返回第一步方向
            if x == target_x and y == target_y:
                return path[0] if path else None

            # 檢查每個方向
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                # 檢查新位置是否有效、可通行且未訪問
                if (maze.xy_valid(new_x, new_y) and
                    maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN] and
                    (new_x, new_y) not in visited):
                    # 標記為已訪問
                    visited.add((new_x, new_y))
                    # 添加新路徑到隊列
                    new_path = path + [(dx, dy)]
                    queue.append((new_x, new_y, new_path))

        # 若無直接路徑，尋找最近的可通行點
        nearby_targets = [
            (new_x, new_y) for dx, dy in directions
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
        ]
        if nearby_targets:
            # 選擇距離目標最近的點
            target = min(nearby_targets, key=lambda p: (p[0] - target_x) ** 2 + (p[1] - target_y) ** 2)
            # 遞迴尋找替代目標的路徑
            return self.bfs_path(start_x, start_y, target[0], target[1], maze)
        return None

    def move_to_target(self, target_x: int, target_y: int, maze) -> bool:
        """
        嘗試移動到目標位置，若無路徑則嘗試附近目標。

        原理：
        - 使用 BFS 尋找目標路徑，確保移動有效。
        - 目標坐標限制在迷宮範圍內，防止越界。
        - 若成功設置新目標，更新 last_x 和 last_y，返回 True。

        Args:
            target_x (int): 目標 x 坐標。
            target_y (int): 目標 y 坐標。
            maze: 迷宮物件。

        Returns:
            bool: 是否成功設置目標並移動。
        """
        # 限制目標坐標在迷宮範圍內
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        # 使用 BFS 尋找路徑
        direction = self.bfs_path(self.x, self.y, target_x, target_y, maze)
        if direction and self.set_new_target(direction[0], direction[1], maze):
            # 更新上次位置，防止反覆移動
            self.last_x, self.last_y = self.x, self.y
            return True
        return False

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        基礎追逐邏輯，子類需覆寫實現具體策略。

        原理：
        - 提供一個虛方法，由子類實現特定的追逐策略（例如直接追逐、包抄、圍堵）。
        - 允許鬼魂根據 Pac-Man 位置和迷宮結構制定智能移動計劃。
        """
        pass

    def return_to_spawn(self, maze):
        """
        快速返回最近的重生點 'S'，優先選擇最短路徑。

        原理：
        - 被 Pac-Man 吃掉後，鬼魂以較高速度（return_speed）返回迷宮中的 'S' 格子。
        - 使用 BFS 尋找最短路徑，若無路徑則隨機移動。
        - 到達重生點後進入等待狀態。

        Args:
            maze: 迷宮物件。
        """
        # 使用返回速度
        self.speed = self.return_speed
        # 收集所有重生點
        spawn_points = [(x, y) for y in range(maze.height)
                        for x in range(maze.width) if maze.get_tile(x, y) == TILE_GHOST_SPAWN]
        if not spawn_points:
            # 如果沒有重生點，隨機移動
            self.move_random(maze)
            return

        # 選擇最近的重生點
        closest_spawn = min(spawn_points, key=lambda p: (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2)
        if self.move_to_target(closest_spawn[0], closest_spawn[1], maze):
            # 如果到達重生點，進入等待狀態
            if self.x == closest_spawn[0] and self.y == closest_spawn[1]:
                self.returning_to_spawn = False
                self.set_waiting(60)  # 假設 fps = 60
            return
        # 如果無法到達，隨機移動
        self.move_random(maze)

    def escape_from_pacman(self, pacman, maze):
        """
        在可食用狀態下逃離 Pac-Man，選擇遠離 Pac-Man 並優先通往高連通性路口的路徑。

        原理：
        - 優先選擇與 Pac-Man 曼哈頓距離最遠的方向，考慮 Pac-Man 的移動方向以避開其預測路徑。
        - 若多個方向距離相近，優先選擇通往路口（連通性 > 2 的格子）的方向。
        - 若無有效方向，執行安全的隨機移動，優先避免靠近 Pac-Man。
        - 使用曼哈頓距離以降低計算開銷，適合網格迷宮。

        Args:
            pacman: PacMan 物件，包含位置 (x, y) 和目標 (target_x, target_y)。
            maze: 迷宮物件，提供 xy_valid 和 get_tile 方法。
        """
        # 定義四個移動方向：下、上、右、左
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 定義可通行格子類型
        valid_tiles = {TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR}
        
        # 預測 Pac-Man 的未來位置（基於目標方向）
        pacman_dx = pacman.target_x - pacman.x if pacman.target_x != pacman.x else 0
        pacman_dy = pacman.target_y - pacman.y if pacman.target_y != pacman.y else 0
        pacman_future = (pacman.x + pacman_dx * 2, pacman.y + pacman_dy * 2)

        best_direction = None
        max_score = -float('inf')
        valid_directions = []

        # 檢查每個方向
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            # 檢查新位置是否有效、可通行且非反覆移動（90% 概率）
            if (maze.xy_valid(new_x, new_y) and 
                maze.get_tile(new_x, new_y) in valid_tiles and
                not (new_x == self.last_x and new_y == self.last_y and random.random() < 0.9)):
                # 計算曼哈頓距離到 Pac-Man 未來位置
                distance = abs(new_x - pacman_future[0]) + abs(new_y - pacman_future[1])
                # 計算連通性分數（可通行方向數）
                connectivity = sum(
                    maze.xy_valid(new_x + ddx, new_y + ddy) and 
                    maze.get_tile(new_x + ddx, new_y + ddy) in valid_tiles
                    for ddx, ddy in directions
                )
                # 綜合評分：距離 + 連通性加權
                score = distance + connectivity * 2
                valid_directions.append((dx, dy, score))
                if score > max_score:
                    max_score = score
                    best_direction = (dx, dy)

        # 如果找到最佳方向，設置新目標
        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            self.last_x, self.last_y = self.x, self.y
            return

        # 從有效方向中選擇，按分數排序
        if valid_directions:
            valid_directions.sort(key=lambda x: x[2], reverse=True)
            for dx, dy, _ in valid_directions:
                if self.set_new_target(dx, dy, maze):
                    self.last_x, self.last_y = self.x, self.y
                    return

        # 若無有效方向，隨機移動
        self.move_random(maze)

    def move_random(self, maze):
        """
        隨機選擇一個可通行方向移動，避免反覆移動。

        原理：
        - 當無明確目標時，隨機選擇可通行方向，優先避免回到上一步位置（90% 概率）。
        - 隨機化移動增加鬼魂行為的多樣性，模擬不確定性。

        Args:
            maze: 迷宮物件。
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.set_new_target(dx, dy, maze):
                return

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """
        設置新目標格子，檢查可通行性並避免反覆移動。

        原理：
        - 檢查新目標 (new_x, new_y) 是否有效且可通行。
        - 若新目標與上一步位置相同，則以 90% 概率拒絕，防止來回移動。
        - 成功設置目標後更新 target_x 和 target_y。

        Args:
            dx (int): x 方向偏移。
            dy (int): y 方向偏移。
            maze: 迷宮物件。

        Returns:
            bool: 是否成功設置目標。
        """
        new_x, new_y = self.x + dx, self.y + dy
        # 檢查是否反覆移動
        if (self.last_x is not None and new_x == self.last_x and new_y == self.last_y and
            random.random() < 0.9):
            return False
        # 檢查新目標是否有效且可通行
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False

    def set_edible(self, duration: int):
        """
        設置鬼魂為可食用狀態，降低移動效率。

        原理：
        - 當 Pac-Man 吃到能量球時，鬼魂進入可食用狀態，持續指定幀數（duration）。
        - 僅在非返回重生點或非等待狀態下生效。

        Args:
            duration (int): 可食用狀態的持續幀數。
        """
        if not (self.returning_to_spawn or self.waiting):
            self.edible = True
            self.edible_timer = duration

    def set_returning_to_spawn(self, fps: int):
        """
        設置鬼魂返回重生點，增加速度並更新狀態。

        原理：
        - 當鬼魂被 Pac-Man 吃掉時，進入返回重生點狀態，速度設為 return_speed。
        - 死亡次數遞增，影響後續速度和等待時間。
        - 重置可食用狀態和透明度。

        Args:
            fps (int): 每秒幀數。
        """
        self.speed = self.return_speed
        self.death_count += 1
        self.returning_to_spawn = True
        self.edible = False
        self.edible_timer = 0
        self.alpha = 255

    def set_waiting(self, fps: int):
        """
        設置鬼魂為等待狀態，根據死亡次數調整等待時間。

        原理：
        - 鬼魂到達重生點後進入等待狀態，等待時間隨死亡次數減少。
        - 等待時間計算公式：wait_time = GHOST_WAIT_TIME / max(1, death_count * 1.5) * (900 / fps)。

        Args:
            fps (int): 每秒幀數。
        """
        self.returning_to_spawn = False
        self.waiting = True
        wait_time = GHOST_WAIT_TIME / max(1, self.death_count * 1.5)
        self.wait_timer = int(wait_time * 900 / fps)

    def reset(self, maze: Map):
        """
        重置鬼魂狀態，恢復初始設置。

        原理：
        - 重置鬼魂的所有狀態屬性，隨機選擇一個重生點作為新位置。
        - 用於遊戲重置或新回合開始。

        Args:
            maze: 迷宮物件。
        """
        self.default_speed = GHOST_DEFAULT_SPEED
        self.speed = self.default_speed
        self.edible = False
        self.edible_timer = 0
        self.returning_to_spawn = False
        self.return_speed = GHOST_RETURN_SPEED
        self.death_count = 0
        self.waiting = False
        self.wait_timer = 0
        self.alpha = 255
        self.last_x = None
        self.last_y = None
        self.memory_x = self.x
        self.memory_y = self.y
        # 隨機選擇一個重生點
        spawn_points = [(x, y) for y in range(maze.height)
                        for x in range(maze.width) if maze.get_tile(x, y) == TILE_GHOST_SPAWN]
        if spawn_points:
            self.x, self.y = random.choice(spawn_points)
        self.target_x = self.x
        self.target_y = self.y

# 子類定義
class Ghost1(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost1"):
        """
        初始化 Ghost1（紅色鬼魂），作為領頭追逐者。

        原理：
        - Ghost1 採用直接追逐策略，預測 Pac-Man 的未來位置。
        - 使用紅色，速度為 GHOST1_SPEED。
        """
        super().__init__(x, y, name, color=RED)
        self.speed = GHOST1_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        領頭追逐策略：預測 Pac-Man 未來 1 步位置，優先使用 BFS 追逐，若無路徑則協調其他鬼魂。

        原理：
        - 預測 Pac-Man 的移動方向，計算未來 1 步位置：target_x = pacman.target_x
        - 使用 BFS 尋找最短路徑，若失敗則追蹤記憶位置或隨機移動。
        - 與其他鬼魂協作，通知最近的鬼魂進行包抄。
        """
        # 追逐 Pac-Man 的目標位置
        if pacman.target_x and pacman.target_y:
            target_x = pacman.target_x
            target_y = pacman.target_y
            target_x = max(0, min(maze.width - 1, target_x))
            target_y = max(0, min(maze.height - 1, target_y))
            if self.move_to_target(target_x, target_y, maze):
                # 更新記憶位置
                self.memory_x, self.memory_y = pacman.x, pacman.y
                return

        # 追逐記憶中的 Pac-Man 位置
        if self.move_to_target(self.memory_x, self.memory_y, maze):
            return

        # 協調其他鬼魂進行包抄
        if ghosts:
            closest_ghost = min(ghosts, key=lambda g: g != self and ((g.x - pacman.x) ** 2 + (g.y - pacman.y) ** 2) ** 0.5, default=None)
            if closest_ghost:
                flank_x = pacman.x + (pacman.x - self.x)
                flank_y = pacman.y + (pacman.y - self.y)
                closest_ghost.move_to_target(flank_x, flank_y, maze)
        # 隨機移動
        self.move_random(maze)
    
    def escape_from_pacman(self, pacman, maze):
        """
        可食用狀態下快速死亡以重新追擊。

        原理：
        - Ghost1 在可食用狀態下不逃跑，而是繼續追逐，模擬快速被吃以重新進入正常狀態。
        """
        self.chase_pacman(pacman, maze)

class Ghost2(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost2"):
        """
        初始化 Ghost2（粉紅色鬼魂），作為側翼包抄者。

        原理：
        - Ghost2 採用包抄策略，與 Ghost1 協作，阻塞 Pac-Man 的逃跑路線。
        - 使用粉紅色，速度為 GHOST2_SPEED。
        """
        super().__init__(x, y, name, color=PINK)
        self.speed = GHOST2_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        側翼包抄策略：根據 Ghost1 位置計算包抄點，阻塞 Pac-Man 逃跑路線。

        原理：
        - 計算與 Ghost1 的對稱點：target_x = pacman.x + (pacman.x - ghost1.x) * 2
        - 若無對稱點，則阻塞 Pac-Man 附近的關鍵路口。
        - 若失敗，則隨機移動。
        """
        # 尋找 Ghost1
        ghost1 = next((g for g in ghosts if g.name == "Ghost1"), None) if ghosts else None
        if not ghost1:
            if self.move_to_target(pacman.x, pacman.y, maze):
                return
            self.move_random(maze)
            return
        
        # 如果距離 Pac-Man 較近，直接追逐
        distance = ((self.x - pacman.x) ** 2 + (self.y - pacman.y) ** 2) ** 0.5
        threshold = 4
        if distance <= threshold:
            if pacman.target_x and pacman.target_y:
                target_x = pacman.target_x
                target_y = pacman.target_y
                target_x = max(0, min(maze.width - 1, target_x))
                target_y = max(0, min(maze.height - 1, target_y))
                if self.move_to_target(target_x, target_y, maze):
                    return
                
        # 計算包抄點
        dx = pacman.x - ghost1.x
        dy = pacman.y - ghost1.y
        target_x = pacman.x + dx * 2
        target_y = pacman.y + dy * 2
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        if self.move_to_target(target_x, target_y, maze):
            return

        # 選擇附近的關鍵路口
        nearby_points = [
            (pacman.x + dx, pacman.y + dy) for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if maze.xy_valid(pacman.x + dx, pacman.y + dy) and
            maze.get_tile(pacman.x + dx, pacman.y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
        ]
        if nearby_points:
            target = max(nearby_points, key=lambda p: ((p[0] - pacman.target_x) ** 2 + (p[1] - pacman.target_y) ** 2) ** 0.5)
            if self.move_to_target(target[0], target[1], maze):
                return
        self.move_random(maze)

class Ghost3(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost3"):
        """
        初始化 Ghost3（青色鬼魂），作為圍堵者。

        原理：
        - Ghost3 採用圍堵策略，與 Ghost1 和 Ghost2 協作，設置陷阱阻塞 Pac-Man。
        - 使用青色，速度為 GHOST3_SPEED。
        """
        super().__init__(x, y, name, color=CYAN)
        self.speed = GHOST3_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        圍堵策略：計算 Pac-Man 可能的逃跑路徑，設置陷阱。

        原理：
        - 預測 Pac-Man 移動方向，計算中點並延長：target_x = (pacman.x + ghost1.x) // 2 + dx * 4
        - 若目標太近，則選擇路口作為陷阱點（路口定義為有多於兩個可通行方向的格子）。
        - 若失敗，則隨機移動。
        """
        ghost1 = next((g for g in ghosts if g.name == "Ghost1"), None) if ghosts else None
        ghost2 = next((g for g in ghosts if g.name == "Ghost2"), None) if ghosts else None
        if not ghost1 or not ghost2:
            if self.move_to_target(pacman.x, pacman.y, maze):
                return
            self.move_random(maze)
            return

        dx, dy = 0, 0
        if pacman.target_x > pacman.x:
            dx = 1
        elif pacman.target_x < pacman.x:
            dx = -1
        elif pacman.target_y > pacman.y:
            dy = 1
        elif pacman.target_y < pacman.y:
            dy = -1

        mid_x = (pacman.x + ghost1.x) // 2
        mid_y = (pacman.y + ghost1.y) // 2
        target_x = mid_x + dx * 4
        target_y = mid_y + dy * 4
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))

        if ((target_x - pacman.x) ** 2 + (target_y - pacman.y) ** 2) ** 0.5 < 6:
            junctions = [
                (x, y) for x, y in [(pacman.x + dx * 2, pacman.y + dy * 2),
                                    (pacman.x + dy, pacman.y - dx),
                                    (pacman.x - dy, pacman.y + dx)]
                if maze.xy_valid(x, y) and maze.get_tile(x, y) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                and sum(maze.xy_valid(x + ddx, y + ddy) and maze.get_tile(x + ddx, y + ddy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                        for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]) > 2
            ]
            if junctions:
                target = min(junctions, key=lambda p: ((p[0] - pacman.x) ** 2 + (p[1] - pacman.y) ** 2) ** 0.5)
                if self.move_to_target(target[0], target[1], maze):
                    return

        if self.move_to_target(target_x, target_y, maze):
            return
        self.move_random(maze)

class Ghost4(Ghost):
    def __init__(self, x: int, y: int, name: str = "Ghost4"):
        """
        初始化 Ghost4（淺藍色鬼魂），作為攪亂者。

        原理：
        - Ghost4 採用攪亂策略，隨機阻塞路口或追逐 Pac-Man。
        - 使用淺藍色，速度為 GHOST4_SPEED。
        """
        super().__init__(x, y, name, color=LIGHT_BLUE)
        self.speed = GHOST4_SPEED

    def chase_pacman(self, pacman, maze, ghosts: List['Ghost'] = None):
        """
        攪亂策略：當距離 Pac-Man 小於 6 格時隨機阻塞路口，否則預測並追逐。

        原理：
        - 若與 Pac-Man 距離小於 6，隨機選擇附近路口阻塞（路口定義為有多於一個可通行方向的格子）。
        - 否則預測 Pac-Man 未來 2 步位置進行追逐。
        - 若失敗，則隨機移動。
        """
        distance = ((self.x - pacman.x) ** 2 + (self.y - pacman.y) ** 2) ** 0.5
        threshold = 6
        if distance < threshold:
            nearby_points = [
                (self.x + dx, self.y + dy)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                if maze.xy_valid(self.x + dx, self.y + dy) and
                maze.get_tile(self.x + dx, self.y + dy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
            ]
            junctions = [
                (x, y) for x, y in nearby_points
                if sum(maze.xy_valid(x + ddx, y + ddy) and maze.get_tile(x + ddx, y + ddy) in [TILE_PATH, TILE_DOOR, TILE_POWER_PELLET, TILE_GHOST_SPAWN]
                       for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]) > 1
            ]
            if junctions:
                target = random.choice(junctions)
                if self.move_to_target(target[0], target[1], maze):
                    return
            self.move_random(maze)
            return

        dx, dy = 0, 0
        if pacman.target_x != pacman.x or pacman.target_y != pacman.y:
            dx = pacman.target_x - pacman.x
            dy = pacman.target_y - pacman.y
        target_x = pacman.x + dx * 2
        target_y = pacman.y + dy * 2
        target_x = max(0, min(maze.width - 1, target_x))
        target_y = max(0, min(maze.height - 1, target_y))
        if self.move_to_target(target_x, target_y, maze):
            return
        if self.move_to_target(pacman.x, pacman.y, maze):
            return
        self.move_random(maze)