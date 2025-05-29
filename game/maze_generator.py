# game/maze_generator.py
"""
迷宮生成模組，負責生成隨機的 Pac-Man 迷宮，包含牆壁、路徑、能量球和鬼魂重生點。
使用隨機牆壁擴展和路徑縮窄演算法，確保迷宮連通且具有挑戰性。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
class Map:
    def __init__(self, w, h, seed=None):
        """
        初始化迷宮，設置尺寸和隨機種子。
        
        Args:
            w (int): 迷宮寬度。
            h (int): 迷宮高度。
            seed (int): 隨機種子，確保可重現的迷宮。
        """
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        self.width = w
        self.w = w
        self.height = h
        self.h = h
        self.tiles = ['.' for _ in range(self.width * self.height)]  # 初始化為路徑
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右方向
        self._initialize_map()

    def _initialize_map(self):
        """初始化迷宮邊界，設置為 '#'。"""
        for x in range(self.width):
            self.set_tile(x, 0, '#')
            self.set_tile(x, self.height - 1, '#')
        for y in range(self.height):
            self.set_tile(0, y, '#')
            self.set_tile(self.width - 1, y, '#')

    def add_central_room(self):
        """
        在迷宮中央添加 5x5 的房間，包含鬼魂重生點 'S' 和門 'D'。
        """
        room = [
            "E...E",
            ".XDX.",
            ".DSD.",
            ".XDX.",
            "E...E"
        ]
        room_w, room_h = 5, 5
        start_x = (self.width - room_w) // 2
        start_y = (self.height - room_h) // 2
        
        if start_x < 1 or start_x + room_w > self.width - 1 or start_y < 1 or start_y + room_h > self.height - 1:
            print("錯誤：迷宮尺寸過小，無法插入中央房間")
            sys.exit(1)
        
        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)
        
        # # 添加房間入口
        # entrances = [
        #     (start_x + 3, start_y - 1),
        #     (start_x + 3, start_y + room_h),
        #     (start_x - 1, start_y + 2),
        #     (start_x + room_w, start_y + 2)
        # ]
        # for ex, ey in entrances:
        #     if self.xy_valid(ex, ey):
        #         self.set_tile(ex, ey, '.')

    def __str__(self):
        """
        返回迷宮的字串表示。
        
        Returns:
            str: 迷宮的字串表示。
        """
        s = ""
        for y in range(self.height):
            for x in range(self.width):
                s += self.tiles[self.xy_to_i(x, y)]
            s += "\n"
        return s

    def xy_to_i(self, x, y):
        """
        將 (x, y) 座標轉換為一維索引。
        
        Returns:
            int: 一維索引。
        """
        return x + y * self.width

    def i_to_xy(self, i):
        """
        將一維索引轉換為 (x, y) 座標。
        
        Returns:
            Tuple: (x, y) 座標。
        """
        return i % self.width, i // self.width

    def xy_valid(self, x, y):
        """
        檢查座標是否有效。
        
        Returns:
            bool: 是否在迷宮範圍內。
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, x, y):
        """
        獲取指定座標的圖塊。
        
        Returns:
            str or None: 圖塊類型或 None（如果無效座標）。
        """
        if not self.xy_valid(x, y):
            return None
        return self.tiles[self.xy_to_i(x, y)]

    def set_tile(self, x, y, value):
        """
        設置指定座標的圖塊類型。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            value (str): 圖塊類型（例如 '#', 'X', '.'）。
        """
        if self.xy_valid(x, y):
            self.tiles[self.xy_to_i(x, y)] = value

    def _flood_fill(self, start_x, start_y, tile_type):
        """
        使用洪水填充演算法計算連通區域的大小和格子。
        
        Args:
            start_x (int): 起始 x 座標。
            start_y (int): 起始 y 座標。
            tile_type (str): 填充的圖塊類型。
        
        Returns:
            Tuple: (連通區域大小, 訪問的格子集合)。
        """
        if self.get_tile(start_x, start_y) != tile_type:
            return 0, set()
        stack = [(start_x, start_y)]
        visited = set([(start_x, start_y)])
        count = 1
        while stack:
            x, y = stack.pop()
            for dx, dy in self.directions:
                new_x, new_y = x + dx, y + dy
                if (self.xy_valid(new_x, new_y) and (new_x, new_y) not in visited and 
                    self.get_tile(new_x, new_y) == tile_type):
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    count += 1
        return count, visited

    def _check_surrounding_paths(self, x, y):
        """
        檢查 (x, y) 周圍九宮格是否全為路徑（不包括自己）。
        
        Returns:
            bool: 是否全為路徑。
        """
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) != '.':
                    return False
        return True

    def _check_dead_end_in_neighborhood(self, x, y):
        """
        檢查 (x, y) 九宮格內是否有不連通的牆壁，可能導致死路。
        
        Returns:
            bool: 是否可能形成死路。
        """
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny):
                    if self.if_dead_end(nx, ny) or self.get_tile(nx, ny) == 'X':
                        return False
        return True

    def _get_connected_wall_size(self, x, y):
        """
        計算 (x, y) 所在連通牆壁組的大小。
        
        Returns:
            int: 連通牆壁的大小。
        """
        size, _ = self._flood_fill(x, y, 'T')
        return size

    def valid_wall_spawnpoint(self, x, y):
        """
        檢查 (x, y) 是否為有效的牆壁生成點。
        
        Returns:
            bool: 是否有效。
        """
        return (self.xy_valid(x, y) and self._check_surrounding_paths(x, y) and 
                self.get_tile(x, y) == '.')

    def convert_all_T_and_A_to_wall(self):
        """將所有 'T' 圖塊轉換為牆壁 'X'。"""
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) in ['T', 'A']:
                    self.set_tile(x, y, 'X')

    def extend_walls(self, extend_prob=0.99):
        """
        以指定概率在現有牆壁的上下左右生成新牆壁，確保不產生死路。
        
        Args:
            extend_prob (float): 擴展牆壁的概率。
        """
        half_width = self.width // 2
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            attempts += 1
            wall_positions = [(x, y) for y in range(1, self.height - 1) 
                             for x in range(1, half_width + 1) if self.valid_wall_spawnpoint(x, y)]
            if not wall_positions:
                break
            
            x, y = random.choice(wall_positions)
            self.set_tile(x, y, 'X')
            if random.random() > extend_prob:
                continue
            direction = random.choice(self.directions)
            new_x, new_y = x + direction[0], y + direction[1]
            tries = 1
            connected_size = 1
            while True:
                self.set_tile(new_x, new_y, 'T')
                if self._check_dead_end_in_neighborhood(new_x, new_y):
                    connected_size += 1
                    if connected_size > 3:
                        break
                    if random.random() < (extend_prob / (1 + (tries)/10)):
                        direction = random.choice(self.directions)
                        new_x, new_y = new_x + direction[0], new_y + direction[1]
                        tries += 1
                    else:
                        break
                elif tries <= 10:
                    self.set_tile(new_x, new_y, '.')
                    direction = random.choice(self.directions)
                    new_x, new_y = x + direction[0], y + direction[1]
                    tries += 1
                else:
                    self.set_tile(new_x, new_y, '.')
                    break

    def if_dead_end(self, x, y):
        """
        檢查 (x, y) 是否為死路（三面或以上被牆壁包圍）。
        
        Returns:
            bool: 是否為死路。
        """
        if not self.xy_valid(x, y) or self.get_tile(x, y) != '.':
            return False
        return sum(1 for dx, dy in self.directions if self.get_tile(x + dx, y + dy) != '.') >= 3

    def _check_connectivity(self, area, blocked_cell=None):
        """
        檢查區域內格子是否連通，允許臨時阻擋一個格子。
        
        Args:
            area (set): 要檢查連通性的格子集合。
            blocked_cell (tuple): 臨時阻擋的格子座標。
        
        Returns:
            bool: 區域是否連通。
        """
        if not area:
            return True
        start = list(area)[0]
        if blocked_cell:
            original_tile = self.get_tile(*blocked_cell)
            self.set_tile(*blocked_cell, 'X')
        _, reachable = self._flood_fill(*start, '.')
        if blocked_cell:
            self.set_tile(*blocked_cell, original_tile)
        return all(pos in reachable for pos in area)

    def narrow_paths(self):
        """
        縮窄 2x2 的路徑區域，隨機嘗試在四個點放置牆壁 'X'，檢查是否在九宮格內形成死路。
        若放置 'X' 後九宮格內的路徑點變為死路，則放置成功；若無死路，則取消放置。

        Returns:
            int: 成功放置的牆壁數量。
        """
        count = 0
        S = 'A'  # 臨時牆壁標記，後續統一轉換為 'X'

        # 遍歷迷宮，尋找 2x2 的路徑區域
        for y in range(1, self.height - 2):
            for x in range(1, self.width - 2):
                # 檢查是否為 2x2 路徑塊（全為 '.'）
                if not (self.get_tile(x, y) == '.' and self.get_tile(x + 1, y) == '.' and
                        self.get_tile(x, y + 1) == '.' and self.get_tile(x + 1, y + 1) == '.'):
                    continue
                
                # 定義 2x2 區域的四個點
                block = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                random.shuffle(block)  # 隨機打亂四個點的順序

                # 嘗試在每個點放置 'X'
                placed = False
                for bx, by in block:
                    self.set_tile(bx, by, S)  # 放置臨時牆壁

                    # 檢查九宮格內的路徑點是否變成死路
                    dead_end_created = False
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = bx + dx, by + dy
                            if self.xy_valid(nx, ny) and self.get_tile(nx, ny) == '.':
                                if self.if_dead_end(nx, ny):
                                    dead_end_created = True
                                    break
                        if dead_end_created:
                            break

                    if not dead_end_created:
                        # 放置成功，保留 'X' 並計數
                        placed = True
                        count += 1
                        break
                    else:
                        # 放置失敗，取消放置（恢復為 '.'）
                        self.set_tile(bx, by, '.')

                # 如果未成功放置，則該 2x2 區域不放置牆壁
                if not placed:
                    continue

        return count

    def place_power_pellets(self):
        """
        在迷宮中均勻放置能量球（'E'），數量根據空地數量動態調整，確保分佈均勻。

        Returns:
            int: 放置的能量球數量。
        """
        if self.seed is not None:
            random.seed(self.seed + 1000)

        # Collect available empty cells, excluding spawn point and its immediate surroundings
        empty_cells = []
        exclude_cells = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == 'S':
                    exclude_cells.add((x, y))
                    for dx, dy in self.directions:
                        nx, ny = x + dx, y + dy
                        if self.xy_valid(nx, ny):
                            exclude_cells.add((nx, ny))

        all_candidate_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == '.' and (x, y) not in exclude_cells:
                    all_candidate_cells.append((x, y))

        empty_count = len(all_candidate_cells)

        # Dynamic pellet count calculation: at least 8, up to 20, scaling with empty cells
        num_pellets = max(8, int(empty_count * 0.1))
        num_pellets = min(num_pellets, 20)

        pellet_positions = set() # Use a set to avoid duplicates

        # If there are enough candidate cells, try to distribute them evenly first
        if empty_count >= num_pellets:
            # Divide the maze into a 4x4 grid and try to place one pellet per sub-grid
            grid_cells_map = {}
            grid_size_x = max(1, self.width // 4)
            grid_size_y = max(1, self.height // 4)

            for cell in all_candidate_cells:
                x, y = cell
                gx = min(x // grid_size_x, 3) # Ensure gx is within 0-3
                gy = min(y // grid_size_y, 3) # Ensure gy is within 0-3
                grid_key = (gx, gy)
                if grid_key not in grid_cells_map:
                    grid_cells_map[grid_key] = []
                grid_cells_map[grid_key].append(cell)

            # Attempt to place one pellet per grid segment first
            for grid_key in random.sample(list(grid_cells_map.keys()), min(len(grid_cells_map), num_pellets)):
                if grid_cells_map[grid_key]:
                    pellet_positions.add(random.choice(grid_cells_map[grid_key]))

            # If not enough pellets placed, or if some sub-grids had no valid cells,
            # select remaining pellets from all available candidate cells
            remaining_needed = num_pellets - len(pellet_positions)
            if remaining_needed > 0:
                available_for_random = [cell for cell in all_candidate_cells if cell not in pellet_positions]
                if available_for_random:
                    pellet_positions.update(random.sample(available_for_random, min(remaining_needed, len(available_for_random))))
        elif empty_count > 0: # Not enough empty cells for the target num_pellets, place as many as possible
            pellet_positions.update(random.sample(all_candidate_cells, empty_count))

        # Place the power pellets on the maze
        for x, y in pellet_positions:
            self.set_tile(x, y, 'E')

        if self.seed is not None:
            random.seed(self.seed)  # Restore original random seed

        return len(pellet_positions)

    def generate_maze(self):
        """
        生成完整的迷宮，包含牆壁擴展、路徑縮窄和能量球放置。
        左半部分生成後鏡像到右半部分，確保對稱性。
        """
        self.extend_walls()
        self.add_central_room()
        while self.narrow_paths():
            pass
        self.convert_all_T_and_A_to_wall()
        self.place_power_pellets()
        half_width = self.width // 2
        for y in range(self.height):
            for x in range(1, half_width):
                self.set_tile(self.width - 1 - x, y, self.get_tile(x, y))
if __name__ == "__main__":
    width, height, seed = MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

    if width < 7 or height < 7:
        print("錯誤：迷宮最小尺寸為 7x7 以容納中央房間")
        sys.exit(1)
    
    maze = Map(width, height, seed=seed)
    maze.generate_maze()
    print(f"生成的Pac-Man迷宮（種子碼：{seed if seed is not None else '無'}）")
    print(f"尺寸：{width}x{height}")
    print(maze)