# game/maze_generator.py
"""
迷宮生成模組，負責生成隨機的 Pac-Man 迷宮，包含牆壁、路徑、能量球和鬼魂重生點。
使用隨機牆壁擴展和路徑縮窄演算法，確保迷宮連通且具有挑戰性。
"""
import random
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

class Map:
    def __init__(self, width, height, seed=None):
        """
        初始化迷宮，設置尺寸和隨機種子。
        
        Args:
            width (int): 迷宮寬度。
            height (int): 迷宮高度。
            seed (int, optional): 隨機種子，確保可重現的迷宮。
        """
        if seed is not None:
            random.seed(seed)
        self.width = width
        self.height = height
        self.tiles = ['.' for _ in range(width * height)]  # 初始化為路徑
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右
        self._initialize_boundaries()

    def _initialize_boundaries(self):
        """設置迷宮邊界為牆壁('#')。"""
        for x in range(self.width):
            self.set_tile(x, 0, '#')
            self.set_tile(x, self.height - 1, '#')
        for y in range(self.height):
            self.set_tile(0, y, '#')
            self.set_tile(self.width - 1, y, '#')

    def add_central_room(self):
        """
        在迷宮中央添加 5x5 房間，包含鬼魂重生點 'S' 和門 'D'。
        檢查尺寸是否足夠，否則退出。
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
            raise ValueError("迷宮尺寸過小，無法插入中央房間")
        
        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)

    def __str__(self):
        """返回迷宮的字串表示。"""
        return '\n'.join(''.join(self.tiles[i:i + self.width]) for i in range(0, len(self.tiles), self.width))

    def xy_to_i(self, x, y):
        """將 (x, y) 座標轉換為一維索引。"""
        return x + y * self.width

    def i_to_xy(self, i):
        """將一維索引轉換為 (x, y) 座標。"""
        return i % self.width, i // self.width

    def xy_valid(self, x, y):
        """檢查座標是否在迷宮範圍內。"""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, x, y):
        """獲取指定座標的圖塊。"""
        return self.tiles[self.xy_to_i(x, y)] if self.xy_valid(x, y) else None

    def set_tile(self, x, y, value):
        """設置指定座標的圖塊類型。"""
        if self.xy_valid(x, y):
            self.tiles[self.xy_to_i(x, y)] = value

    def _flood_fill(self, start_x, start_y, tile_type):
        """使用洪水填充計算連通區域。"""
        if self.get_tile(start_x, start_y) != tile_type:
            return 0, set()
        stack = [(start_x, start_y)]
        visited = {(start_x, start_y)}
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

    def _is_surrounded_by_paths(self, x, y):
        """檢查周圍九宮格是否全為路徑。"""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) != '.':
                    return False
        return True

    def _has_dead_end_in_neighborhood(self, x, y):
        """檢查九宮格內是否可能形成死路。"""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) in ['X', 'T']:
                    return False
        return True

    def _get_connected_wall_size(self, x, y):
        """計算連通牆壁的大小。"""
        return self._flood_fill(x, y, 'T')[0]

    def is_valid_wall_spawnpoint(self, x, y):
        """檢查是否為有效的牆壁生成點。"""
        return (self.xy_valid(x, y) and self._is_surrounded_by_paths(x, y) and 
                self.get_tile(x, y) == '.')

    def convert_temp_walls(self):
        """將臨時 'T' 和 'A' 圖塊轉換為牆壁 'X'。"""
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) in ['T', 'A']:
                    self.set_tile(x, y, 'X')

    def extend_walls(self, extend_prob=0.99):
        """以指定概率擴展牆壁，確保不產生死路。"""
        half_width = self.width // 2
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            wall_positions = [(x, y) for y in range(1, self.height - 1) 
                            for x in range(1, half_width + 1) if self.is_valid_wall_spawnpoint(x, y)]
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
            while tries <= 10:
                self.set_tile(new_x, new_y, 'T')
                if self._has_dead_end_in_neighborhood(new_x, new_y):
                    connected_size += 1
                    if connected_size > 3 or random.random() < (extend_prob / (1 + tries/10)):
                        direction = random.choice(self.directions)
                        new_x, new_y = new_x + direction[0], new_y + direction[1]
                        tries += 1
                    else:
                        break
                else:
                    self.set_tile(new_x, new_y, '.')
                    direction = random.choice(self.directions)
                    new_x, new_y = x + direction[0], y + direction[1]
                    tries += 1
            attempts += 1

    def is_dead_end(self, x, y):
        """檢查是否為死路（三面或以上被牆壁包圍）。"""
        if not self.xy_valid(x, y) or self.get_tile(x, y) != '.':
            return False
        return sum(1 for dx, dy in self.directions if self.get_tile(x + dx, y + dy) != '.') >= 3

    def _check_connectivity(self, area, blocked_cell=None):
        """檢查區域內格子是否連通。"""
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
        """縮窄 2x2 路徑區域，生成死路。"""
        count = 0
        temp_wall = 'A'
        for y in range(1, self.height - 2):
            for x in range(1, self.width - 2):
                if all(self.get_tile(x + dx, y + dy) == '.' for dx in range(2) for dy in range(2)):
                    block = [(x + dx, y + dy) for dx in range(2) for dy in range(2)]
                    random.shuffle(block)
                    placed = False
                    for bx, by in block:
                        self.set_tile(bx, by, temp_wall)
                        dead_end_created = any(self.is_dead_end(nx, ny) for nx, ny in block 
                                            if self.get_tile(nx, ny) == '.')
                        if dead_end_created:
                            placed = True
                            count += 1
                            break
                        self.set_tile(bx, by, '.')
                    if not placed:
                        for bx, by in block:
                            self.set_tile(bx, by, '.')
        return count

    def place_power_pellets(self):
        """均勻放置能量球 'E'。"""
        empty_cells = [(x, y) for y in range(self.height) for x in range(self.width) 
                      if self.get_tile(x, y) == '.' and (x, y) not in [(x, y) for x, y in 
                      [(x, y) for y in range(self.height) for x in range(self.width) 
                      if self.get_tile(x, y) == 'S']]]
        if not empty_cells:
            return 0
        num_pellets = min(max(8, int(len(empty_cells) * 0.1)), 20)
        pellet_positions = random.sample(empty_cells, min(num_pellets, len(empty_cells)))
        for x, y in pellet_positions:
            self.set_tile(x, y, 'E')
        return len(pellet_positions)

    def generate_maze(self):
        """生成完整迷宮，包括牆壁擴展、路徑縮窄和能量球放置。"""
        self.extend_walls()
        self.add_central_room()
        while self.narrow_paths():
            pass
        self.convert_temp_walls()
        self.place_power_pellets()
        half_width = self.width // 2
        for y in range(self.height):
            for x in range(1, half_width):
                self.set_tile(self.width - 1 - x, y, self.get_tile(x, y))

if __name__ == "__main__":
    maze = Map(MAZE_WIDTH, MAZE_HEIGHT, seed=MAZE_SEED)
    maze.generate_maze()
    print(f"生成的 Pac-Man 迷宮（種子碼：{MAZE_SEED if MAZE_SEED else '無'}）")
    print(f"尺寸：{MAZE_WIDTH}x{MAZE_HEIGHT}")
    print(maze)