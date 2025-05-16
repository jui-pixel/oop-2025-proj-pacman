#!/usr/bin/python3
"""
生成一個隨機的Pac-Man迷宮，根據輸入的寬度和高度先生成矩陣，先放置中央房間，
然後通過放置預設牆壁形狀（一格、I、L、T、+）生成迷宮。
滿足以下限制：
- 路徑僅 1 格厚。
- 交叉口之間至少相隔 2 格（曼哈頓距離至少 3）。
- 有 1 或 2 條隧道。
- 無死路。
- 牆壁形狀為 一格、I、L、T、+。
支持種子碼隨機化、可自定義大小和對稱佈局。
障礙物為 'X'，幽靈生成位置為 'S'，包含 7x5 中央房間。
"""

import sys
import random

class Map:
    def __init__(self, width, height, seed=None):
        if seed is not None:
            random.seed(seed)
        self.width = width
        self.height = height
        self.tiles = ['.' for _ in range(self.width * self.height)]
        self.wall_shapes = {
            "single": [(0, 0)],
            "I_horizontal": [(0, 0), (-1, 0), (1, 0)],
            "I_vertical": [(0, 0), (0, -1), (0, 1)],
            "L_top_left": [(0, 0), (1, 0), (0, 1)],
            "L_top_right": [(0, 0), (-1, 0), (0, 1)],
            "L_bottom_left": [(0, 0), (1, 0), (0, -1)],
            "L_bottom_right": [(0, 0), (-1, 0), (0, -1)],
            "T_up": [(0, 0), (-1, 0), (1, 0), (0, 1)],
            "T_down": [(0, 0), (-1, 0), (1, 0), (0, -1)],
            "T_left": [(0, 0), (0, -1), (0, 1), (1, 0)],
            "T_right": [(0, 0), (0, -1), (0, 1), (-1, 0)],
            "plus": [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        }
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self._initialize_map()
        self.add_central_room()

    def _initialize_map(self):
        """初始化迷宮，設置邊界。"""
        for x in range(self.width):
            self.set_tile(x, 0, '#')
            self.set_tile(x, self.height - 1, '#')
        for y in range(self.height):
            self.set_tile(0, y, '#')
            self.set_tile(self.width - 1, y, '#')

    def add_central_room(self):
        """添加中央 7x5 房間。"""
        room = [
            ".......",
            ".......",
            ".XSSSX.",
            ".......",
            "......."
        ]
        room_w, room_h = 7, 5
        start_x = (self.width - room_w) // 2
        start_y = (self.height - room_h) // 2

        if start_x < 1 or start_x + room_w > self.width - 1 or start_y < 1 or start_y + room_h > self.height - 1:
            print("錯誤：迷宮尺寸過小，無法插入中央房間")
            sys.exit(1)

        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)

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

    def _flood_fill(self, start_x, start_y):
        if self.get_tile(start_x, start_y) != '.':
            return 0, set()
        stack = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        count = 1
        while stack:
            x, y = stack.pop()
            for dx, dy in self.directions:
                new_x, new_y = x + dx, y + dy
                if self.xy_valid(new_x, new_y) and (new_x, new_y) not in visited and self.get_tile(new_x, new_y) == '.':
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    count += 1
        return count, visited

    def _is_connected(self):
        half_width = self.width // 2
        start_x, start_y = None, None
        for y in range(1, self.height - 1):
            for x in range(1, half_width + 1):
                if self.get_tile(x, y) == '.':
                    start_x, start_y = x, y
                    break
            if start_x is not None:
                break
        if start_x is None:
            return True
        reachable_count, _ = self._flood_fill(start_x, start_y)
        total_dots = sum(1 for y in range(1, self.height - 1) for x in range(1, half_width + 1)
                         if self.get_tile(x, y) == '.')
        return reachable_count == total_dots

    def _is_intersection(self, x, y):
        open_paths = sum(1 for dx, dy in self.directions if self.get_tile(x + dx, y + dy) == '.')
        return open_paths >= 3

    def _check_path_thickness(self, x, y):
        if self.get_tile(x, y) != '.':
            return True
        adjacent_paths = sum(1 for dx, dy in [(1, 0), (-1, 0)] if self.get_tile(x + dx, y + dy) == '.')
        if adjacent_paths > 1:
            return False
        adjacent_paths = sum(1 for dx, dy in [(0, 1), (0, -1)] if self.get_tile(x + dx, y + dy) == '.')
        if adjacent_paths > 1:
            return False
        return True

    def _check_intersection_distance(self, x, y):
        if not self._is_intersection(x, y):
            return True
        visited = set()
        queue = [(x, y, 0)]
        while queue:
            cx, cy, dist = queue.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            if dist > 0 and self._is_intersection(cx, cy) and dist < 3:
                return False
            if dist >= 3:
                continue
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if self.xy_valid(nx, ny) and (nx, ny) not in visited and self.get_tile(nx, ny) == '.':
                    queue.append((nx, ny, dist + 1))
        return True

    def _check_no_dead_ends(self):
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.get_tile(x, y) == '.':
                    open_paths = sum(1 for dx, dy in self.directions if self.get_tile(x + dx, y + dy) in ['.', 'T'])
                    if open_paths < 2:
                        return False
        return True

    def _can_place_shape(self, shape, center_x, center_y):
        coords = self.wall_shapes[shape]
        for dx, dy in coords:
            x, y = center_x + dx, center_y + dy
            if not self.xy_valid(x, y) or self.get_tile(x, y) in ['#', 'S']:
                return False
        return True

    def _place_shape(self, shape, center_x, center_y):
        coords = self.wall_shapes[shape]
        affected = []
        for dx, dy in coords:
            x, y = center_x + dx, center_y + dy
            affected.append((x, y, self.get_tile(x, y)))
            self.set_tile(x, y, 'X')
        return affected

    def _undo_shape(self, affected):
        for x, y, original in affected:
            self.set_tile(x, y, original)

    def add_obstacles(self, wall_density=0.4):
        """放置障礙物（牆壁形狀）。"""
        shapes = list(self.wall_shapes.keys())
        half_width = self.width // 2
        target_walls = int(half_width * (self.height - 2) * wall_density)
        current_walls = sum(1 for y in range(1, self.height - 1) for x in range(1, half_width + 1)
                           if self.get_tile(x, y) == 'X')
        attempts = 0
        max_attempts = 1000

        while current_walls < target_walls and attempts < max_attempts:
            x = random.randint(1, half_width)
            y = random.randint(1, self.height - 2)
            shape = random.choice(shapes)
            if self._can_place_shape(shape, x, y):
                affected = self._place_shape(shape, x, y)
                valid = True
                for ax, ay, _ in affected:
                    if not self._check_path_thickness(ax, ay) or not self._check_intersection_distance(ax, ay):
                        valid = False
                        break
                if valid and self._is_connected():
                    current_walls = sum(1 for y in range(1, self.height - 1) for x in range(1, half_width + 1)
                                        if self.get_tile(x, y) == 'X')
                else:
                    self._undo_shape(affected)
            attempts += 1

    def remove_dead_ends(self):
        """移除死路。"""
        while True:
            modified = False
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.get_tile(x, y) == '.':
                        open_paths = sum(1 for dx, dy in self.directions if self.get_tile(x + dx, y + dy) in ['.', 'T'])
                        if open_paths < 2:
                            self.set_tile(x, y, 'X')
                            modified = True
            if not modified:
                break

    def add_tunnels(self):
        """添加 1 或 2 條隧道。"""
        half_width = self.width // 2
        possible_positions = [(x, 0) for x in range(2, half_width - 1) if self.get_tile(x, 1) == '.']
        possible_positions.extend((x, self.height - 1) for x in range(2, half_width - 1) if self.get_tile(x, self.height - 2) == '.')
        num_tunnels = min(random.randint(1, 2), len(possible_positions))
        if num_tunnels == 0:
            return
        tunnel_positions = random.sample(possible_positions, num_tunnels)
        for tx, ty in tunnel_positions:
            self.set_tile(tx, ty, 'T')
            self.set_tile(self.width - 1 - tx, ty, 'T')

    def generate_maze(self):
        """生成迷宮，先放置中央房間，再添加障礙物。"""
        self.add_obstacles()
        self.add_tunnels()
        self.remove_dead_ends()

        # 鏡像到右半部分
        half_width = self.width // 2
        for y in range(self.height):
            for x in range(1, half_width):
                self.set_tile(self.width - 1 - x, y, self.get_tile(x, y))
            if self.width % 2 == 1 and self.get_tile(half_width, y) in ['S']:
                self.set_tile(half_width, y, '.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='生成一個連通的Pac-Man迷宮')
    parser.add_argument('--width', type=int, default=29, help='迷宮寬度')
    parser.add_argument('--height', type=int, default=31, help='迷宮高度')
    parser.add_argument('--seed', type=int, help='隨機種子碼，用於可重現的生成')
    args = parser.parse_args()

    if args.width < 9 or args.height < 9:
        print("錯誤：迷宮最小尺寸為 9x9 以容納中央房間")
        sys.exit(1)

    maze = Map(args.width, args.height, seed=args.seed)
    maze.generate_maze()

    print(f"生成的Pac-Man迷宮（種子碼：{args.seed if args.seed is not None else '無'}）")
    print(f"尺寸：{args.width}x{args.height}")
    print(maze)