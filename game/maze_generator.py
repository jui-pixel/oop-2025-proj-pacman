#!/usr/bin/python3
"""
生成一個隨機的Pac-Man迷宮，通過放置和擴展牆壁塊生成。
支持放置階段、成長階段和擴展階段，並確保無死路、單格厚牆。
包含中央房間、隧道和對稱佈局。
"""

import sys
import random

class Obstacle:
    """表示一組連續的牆壁塊。"""
    def __init__(self, x, y, shape):
        self.tiles = [(x + dx, y + dy) for dx, dy in shape]
        self.direction = None  # 擴展方向

    def add_tile(self, x, y):
        """添加新瓦片到障礙物。"""
        self.tiles.append((x, y))

    def get_bounds(self):
        """返回障礙物的邊界（min_x, max_x, min_y, max_y）。"""
        x_coords, y_coords = zip(*self.tiles)
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

class Map:
    def __init__(self, width, height, tile_str=None):
        self.width = width
        self.height = height
        self.tiles = []
        self.obstacles = []  # 儲存所有障礙物
        if tile_str is None:
            self._initialize_empty_map()
        else:
            self._load_from_string(tile_str)
        self.verbose = False

    def _initialize_empty_map(self):
        self.tiles = ['.' for _ in range(self.width * self.height)]
        for x in range(self.width):
            self.set_tile(x, 0, '#')
            self.set_tile(x, self.height - 1, '#')
        for y in range(self.height):
            self.set_tile(0, y, '#')
            self.set_tile(self.width - 1, y, '#')

    def _load_from_string(self, tile_str):
        self.tiles = list(self._format_map_str(tile_str, ""))

    @staticmethod
    def _format_map_str(tiles, sep):
        return sep.join(line.strip() for line in tiles.splitlines())

    def __str__(self):
        s = ""
        for y in range(self.height):
            for x in range(self.width):
                s += self.tiles[self.xy_to_i(x, y)]
            s += "\n"
        return s

    def xy_to_i(self, x, y):
        return x + y * self.width

    def i_to_xy(self, i):
        return i % self.width, i // self.width

    def xy_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, x, y):
        if not self.xy_valid(x, y):
            return None
        return self.tiles[self.xy_to_i(x, y)]

    def set_tile(self, x, y, value):
        if self.xy_valid(x, y):
            self.tiles[self.xy_to_i(x, y)] = value

    def is_wall_block_filled(self, x, y):
        """檢查 (x, y) 開始的 2x2 塊是否完全由牆壁 ('|') 組成。"""
        return all(self.get_tile(x + dx, y + dy) == '|' for dy in range(2) for dx in range(2))

    def can_new_block_fit(self, x, y):
        """檢查 2x2 塊是否能在 4x4 區域內放置。"""
        if not (self.xy_valid(x, y) and self.xy_valid(x + 3, y + 3)):
            return False
        for y0 in range(y, y + 4):
            for x0 in range(x, x + 4):
                if self.get_tile(x0, y0) != '.':
                    return False
        return True

    def update_pos_list(self):
        """更新可放置位置列表。"""
        self.pos_list = [(x, y) for y in range(self.height - 3) for x in range(self.width - 3)
                         if self.can_new_block_fit(x, y)]

    def add_connection(self, x, y, dx, dy):
        """添加連接關係，確保相鄰牆壁填充。"""
        if (x, y) in self.pos_list:
            for i in range(4):
                x0, y0 = x + dx + i * dx, y + dy + i * dy
                if self.xy_valid(x0, y0) and (x0, y0) in self.pos_list:
                    if (x0, y0) in self.connections:
                        self.connections[(x0, y0)].append((x, y))
                    else:
                        self.connections[(x0, y0)] = [(x, y)]

    def update_connections(self):
        """更新所有位置的連接關係。"""
        self.connections = {}
        for y in range(self.height - 3):
            for x in range(self.width - 3):
                if (x, y) in self.pos_list:
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        if any(self.get_tile(x + dx * 4, y + dy * i) == '|' for i in range(4)):
                            self.add_connection(x, y, dx, dy)

    def update(self):
        self.update_pos_list()
        self.update_connections()

    def add_wall_block(self, x, y):
        """添加 2x2 牆壁塊並創建新障礙物。"""
        shape = [(0, 0), (1, 0), (0, 1), (1, 1)]
        obstacle = Obstacle(x, y, shape)
        for dx, dy in shape:
            self.set_tile(x + dx, y + dy, '|')
            obstacle.add_tile(x + dx, y + dy)
        self.obstacles.append(obstacle)

    def expand_wall(self, obstacle):
        """擴展牆壁塊，填充相鄰區域。"""
        visited = set()
        count = 0
        stack = [(obstacle.tiles[0][0], obstacle.tiles[0][1])]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x0, y0 = x + dx, y + dy
                if self.xy_valid(x0, y0) and self.get_tile(x0, y0) == '.':
                    pos = (x0 - 1, y0 - 1)
                    if pos in self.pos_list and pos in self.connections:
                        for conn_x, conn_y in self.connections[pos]:
                            if not self.is_wall_block_filled(conn_x, conn_y):
                                self.add_wall_block(conn_x, conn_y)
                                obstacle.add_tile(x0, y0)
                                count += 1
                                stack.append((x0, y0))
        return count

    def extend_wall(self, obstacle, max_blocks=4):
        """沿隨機方向擴展牆壁。"""
        if not obstacle.tiles:
            return 0
        x, y = obstacle.tiles[0]
        dx, dy = random.choice([(0, -1), (0, 1), (1, 0), (-1, 0)])
        orig_dir = (dx, dy)
        turn = False
        turn_blocks = random.randint(2, 4) if random.random() <= 0.35 else max_blocks
        count = 0
        i = 0

        while count < max_blocks:
            x0, y0 = x + dx * i, y + dy * i
            pos = (x0 - 1, y0 - 1)
            if not self.xy_valid(x0, y0) or pos not in self.pos_list or turn and orig_dir == (dx, dy):
                break
            if not turn and (count >= turn_blocks or not self.can_new_block_fit(x0 - 1, y0 - 1)):
                turn = True
                dx, dy = -dy, dx
                i = 1
                continue
            if not self.is_wall_block_filled(x0 - 1, y0 - 1):
                self.add_wall_block(x0 - 1, y0 - 1)
                count += 1 + self.expand_wall(obstacle)
            i += 1
        return count

    def add_wall_obstacle(self, x=None, y=None, extend=False):
        self.update()
        if not self.pos_list:
            return False

        if x is None or y is None:
            x, y = random.choice(self.pos_list)

        self.add_wall_block(x, y)
        obstacle = self.obstacles[-1]
        count = self.expand_wall(obstacle)

        if extend:
            count += self.extend_wall(obstacle)

        if self.verbose:
            print(f"Added obstacle at ({x}, {y}), expanded: {count}")
            print(self)

        return True

    def remove_dead_ends(self):
        """移除死路和單格厚牆。"""
        changed = True
        while changed:
            changed = False
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.get_tile(x, y) == '.':
                        open_paths = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                       if self.get_tile(x + dx, y + dy) in ['.', 'T'])
                        if open_paths < 2:
                            self.set_tile(x, y, 'X')
                            changed = True
                    elif self.get_tile(x, y) == '|' and not self.is_wall_block_filled(x - 1, y - 1):
                        self.set_tile(x, y, 'X')
                        changed = True

    def add_tunnels(self):
        """添加 1 或 2 條隧道。"""
        half_width = self.width // 2
        positions = [(x, 0) for x in range(2, half_width - 1) if self.get_tile(x, 1) == '.']
        positions.extend((x, self.height - 1) for x in range(2, half_width - 1) if self.get_tile(x, self.height - 2) == '.')
        num_tunnels = min(random.randint(1, 2), len(positions))
        if num_tunnels > 0:
            for x, y in random.sample(positions, num_tunnels):
                self.set_tile(x, y, 'T')
                self.set_tile(self.width - 1 - x, y, 'T')

    def add_central_room(self):
        """添加中央 7x5 房間。"""
        room = [
            ".......",
            ".......",
            ".XSSSX.",
            ".......",
            "......."
        ]
        start_x = (self.width - 7) // 2
        start_y = (self.height - 5) // 2

        if start_x < 1 or start_x + 7 > self.width - 1 or start_y < 1 or start_y + 5 > self.height - 1:
            print("錯誤：迷宮尺寸過小，無法插入中央房間")
            sys.exit(1)

        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)

        entrances = [
            (start_x + 3, start_y - 1),
            (start_x + 3, start_y + 5),
            (start_x - 1, start_y + 2),
            (start_x + 7, start_y + 2)
        ]
        for ex, ey in entrances:
            if self.xy_valid(ex, ey):
                self.set_tile(ex, ey, '.')

        while not all(self.get_tile(x, y) in ['.', 'T', '#', 'X', 'S'] for y in range(self.height) for x in range(self.width)):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.get_tile(x, y) == '.':
                        open_paths = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                       if self.get_tile(x + dx, y + dy) in ['.', 'T'])
                        if open_paths < 2:
                            self.set_tile(x, y, 'X')

if __name__ == "__main__":
    tile_map = Map(29, 31, """
        |||||||||||||||||||||||||||||||
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.........||||||||||.........|
        |.........||||||||||.........|
        |.........||||||||||.........|
        |.........||||||||||.........|
        |.........||||||||||.........|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |.............................|
        |||||||||||||||||||||||||||||||
        """)

    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        tile_map.verbose = True

    while tile_map.add_wall_obstacle(extend=True):
        pass

    tile_map.remove_dead_ends()
    tile_map.add_tunnels()
    tile_map.add_central_room()

    half_width = tile_map.width // 2
    for line in str(tile_map).splitlines():
        s = line[:half_width]
        print(s + s[::-1])