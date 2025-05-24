# game/maze_generator.py
"""
迷宮生成模組，負責生成隨機的 Pac-Man 迷宮，包含牆壁、路徑、能量球和鬼魂重生點。
使用隨機牆壁擴展和路徑縮窄演算法，確保迷宮連通且具有挑戰性。
"""
import sys
import random

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
        self.add_central_room()

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
        在迷宮中央添加 7x5 的房間，包含鬼魂重生點 'S' 和門 'D'。
        """
        room = [
            ".....",
            ".XDX.",
            ".DSD.",
            ".XDX.",
            "....."
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
        
        # 添加房間入口
        entrances = [
            (start_x + 3, start_y - 1),
            (start_x + 3, start_y + room_h),
            (start_x - 1, start_y + 2),
            (start_x + room_w, start_y + 2)
        ]
        for ex, ey in entrances:
            if self.xy_valid(ex, ey):
                self.set_tile(ex, ey, '.')

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
                    if self.get_tile(nx, ny) not in ['.', 'T']:
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

    def convert_nearby_T_to_wall(self, x, y):
        """
        將 (x, y) 附近的 'T' 圖塊轉換為牆壁 'X'。
        """
        if self.get_tile(x, y) == 'T':
            _, connected_tiles = self._flood_fill(x, y, 'T')
            for cx, cy in connected_tiles:
                self.set_tile(cx, cy, 'X')

    def convert_all_T_to_wall(self):
        """將所有 'T' 圖塊轉換為牆壁 'X'。"""
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == 'T':
                    self.set_tile(x, y, 'X')

    def extend_walls(self, extend_prob=0.9):
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
            self.set_tile(x, y, 'T')
            if random.random() > extend_prob:
                continue
            direction = random.choice(self.directions)
            new_x, new_y = x + direction[0], y + direction[1]
            
            while self._check_dead_end_in_neighborhood(new_x, new_y):
                self.set_tile(new_x, new_y, 'T')
                connected_size = self._get_connected_wall_size(new_x, new_y)
                if connected_size > 4:
                    break
                if random.random() < extend_prob:
                    direction = random.choice(self.directions)
                    new_x, new_y = new_x + direction[0], new_y + direction[1]
                else:
                    break
            
            self.convert_nearby_T_to_wall(x, y)

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
        縮窄 2x2 空地塊，隨機添加牆壁，確保不產生死路。
        
        Returns:
            int: 放置的牆壁數量。
        """
        count = 0
        S = 'A'  # 臨時牆壁標記
        for y in range(1, self.height - 2):
            for x in range(1, self.width - 2):
                if not (self.get_tile(x, y) == '.' and self.get_tile(x + 1, y) == '.' and
                        self.get_tile(x, y + 1) == '.' and self.get_tile(x + 1, y + 1) == '.'):
                    continue
                
                block = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                neighbors = set()
                for bx, by in block:
                    for dx, dy in self.directions:
                        nx, ny = bx + dx, by + dy
                        if self.xy_valid(nx, ny) and self.get_tile(nx, ny) == '.' and (nx, ny) not in block:
                            neighbors.add((nx, ny))
                
                placed = False
                conditions = [
                    ((x, y), lambda: self.get_tile(x - 1, y) != '.' and self.get_tile(x, y - 1) != '.'),
                    ((x + 1, y), lambda: self.get_tile(x + 2, y) != '.' and self.get_tile(x + 1, y - 1) != '.'),
                    ((x, y + 1), lambda: self.get_tile(x, y + 2) != '.' and self.get_tile(x - 1, y + 1) != '.'),
                    ((x + 1, y + 1), lambda: self.get_tile(x + 2, y + 1) != '.' and self.get_tile(x + 1, y + 2) != '.')
                ]
                random.shuffle(conditions)
                
                for (cx, cy), condition in conditions:
                    if condition() and self._check_connectivity(neighbors, (cx, cy)):
                        self.set_tile(cx, cy, S)
                        dead_end = any(self.if_dead_end(nx, ny) for dx, dy in self.directions 
                                      for nx, ny in [(cx + dx, cy + dy)] if self.xy_valid(nx, ny))
                        if not dead_end:
                            placed = True
                            count += 1
                            break
                        self.set_tile(cx, cy, '.')
                
                if not placed:
                    random.shuffle(block)
                    for bx, by in block:
                        if self._check_connectivity(neighbors, (bx, by)):
                            self.set_tile(bx, by, S)
                            dead_end = any(self.if_dead_end(nx, ny) for dx, dy in self.directions 
                                          for nx, ny in [(bx + dx, by + dy)] if self.xy_valid(nx, ny))
                            if not dead_end:
                                placed = True
                                count += 1
                                break
                            self.set_tile(bx, by, '.')
        
        # 將臨時標記轉為牆壁
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == 'A':
                    self.set_tile(x, y, 'X')
        
        return count

    def place_power_pellets(self):
        """
        在迷宮中均勻放置能量球（'E'），數量根據空地數量動態調整。
        
        Returns:
            int: 放置的能量球數量。
        """
        if self.seed is not None:
            random.seed(self.seed + 1000)
        
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
        
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == '.' and (x, y) not in exclude_cells:
                    empty_cells.append((x, y))
        
        empty_count = len(empty_cells)
        num_pellets = max(4, min(int(empty_count * 0.08), 20))
        
        grid_size_x = self.width // 4
        grid_size_y = self.height // 4
        pellet_positions = []
        pellets_per_grid = max(1, num_pellets // 16)
        
        for gy in range(4):
            for gx in range(4):
                x_start = gx * grid_size_x
                x_end = min((gx + 1) * grid_size_x, self.width)
                y_start = gy * grid_size_y
                y_end = min((gy + 1) * grid_size_y, self.height)
                grid_cells = [(x, y) for x in range(x_start, x_end) for y in range(y_start, y_end)
                             if (x, y) in empty_cells]
                if grid_cells:
                    num_to_place = min(len(grid_cells), pellets_per_grid)
                    selected_cells = random.sample(grid_cells, num_to_place)
                    pellet_positions.extend(selected_cells)
                    empty_cells = [c for c in empty_cells if c not in selected_cells]
        
        remaining = num_pellets - len(pellet_positions)
        if remaining > 0 and empty_cells:
            additional_cells = random.sample(empty_cells, min(remaining, len(empty_cells)))
            pellet_positions.extend(additional_cells)
        
        for x, y in pellet_positions:
            self.set_tile(x, y, 'E')
        
        if self.seed is not None:
            random.seed(self.seed)
        
        return len(pellet_positions)

    def generate_maze(self):
        """
        生成完整的迷宮，包含牆壁擴展、路徑縮窄和能量球放置。
        左半部分生成後鏡像到右半部分，確保對稱性。
        """
        self.extend_walls()
        self.convert_all_T_to_wall()
        
        half_width = self.width // 2
        for y in range(self.height):
            for x in range(1, half_width):
                self.set_tile(self.width - 1 - x, y, self.get_tile(x, y))
        while self.narrow_paths():
            pass
        self.place_power_pellets()

if __name__ == "__main__":
    width = 19
    height = 19
    seed = 1
    
    if width < 7 or height < 7:
        print("錯誤：迷宮最小尺寸為 7x7 以容納中央房間")
        sys.exit(1)
    
    maze = Map(width, height, seed=seed)
    maze.generate_maze()
    print(f"生成的Pac-Man迷宮（種子碼：{seed if seed is not None else '無'}）")
    print(f"尺寸：{width}x{height}")
    print(maze)