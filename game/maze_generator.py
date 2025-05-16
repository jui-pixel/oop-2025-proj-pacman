#!/usr/bin/python3
# game/maze_generator.py
"""
生成一個隨機的Pac-Man迷宮，採用放置預設牆壁形狀（一格、I、L、T、+）的方式，滿足以下限制：
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
import argparse

class BaseMap:
    """基礎映射類，負責迷宮的基礎操作和存儲。"""

    def __init__(self, width, height, seed=None, tile_str=None):
        if seed is not None:
            random.seed(seed)
        self.width = width
        self.height = height
        self.tiles = []
        if tile_str is None:
            self._initialize_empty_map()
        else:
            self._load_from_string(tile_str)

    def _initialize_empty_map(self):
        self.tiles = ['.' for _ in range(self.width * self.height)]  # 初始為全路徑
        for x in range(self.width):
            self.set_tile(x, 0, '#')
            self.set_tile(x, self.height - 1, '#')
        for y in range(self.height):
            self.set_tile(0, y, '#')
            self.set_tile(self.width - 1, y, '#')

    def _load_from_string(self, tile_str):
        formatted_tiles = self._format_map_str(tile_str, "")
        self.tiles = list(formatted_tiles)

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

class MazeGenerator:
    """迷宮生成器，通過放置預設牆壁形狀生成迷宮。"""

    def __init__(self, base_map, config=None):
        self.map = base_map
        self.config = config or {
            "wall_density": 0.4,  # 牆壁密度
            "min_intersection_distance": 3,
            "num_tunnels": random.randint(1, 2),
            "central_room": {
                "width": 7,
                "height": 5,
                "template": [
                    ".......",
                    ".......",
                    ".XSSSX.",
                    ".......",
                    "......."
                ]
            }
        }
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.wall_shapes = {
            "single": [(0, 0)],  # 一格
            "I_horizontal": [(0, 0), (-1, 0), (1, 0)],  # 水平 I 形
            "I_vertical": [(0, 0), (0, -1), (0, 1)],  # 垂直 I 形
            "L_top_left": [(0, 0), (1, 0), (0, 1)],  # L 形（左上）
            "L_top_right": [(0, 0), (-1, 0), (0, 1)],  # L 形（右上）
            "L_bottom_left": [(0, 0), (1, 0), (0, -1)],  # L 形（左下）
            "L_bottom_right": [(0, 0), (-1, 0), (0, -1)],  # L 形（右下）
            "T_up": [(0, 0), (-1, 0), (1, 0), (0, 1)],  # T 形（朝上）
            "T_down": [(0, 0), (-1, 0), (1, 0), (0, -1)],  # T 形（朝下）
            "T_left": [(0, 0), (0, -1), (0, 1), (1, 0)],  # T 形（朝左）
            "T_right": [(0, 0), (0, -1), (0, 1), (-1, 0)],  # T 形（朝右）
            "plus": [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # + 形
        }

    def _flood_fill(self, start_x, start_y, tiles):
        if tiles[self.map.xy_to_i(start_x, start_y)] != '.':
            return 0, set()
        stack = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        count = 1
        while stack:
            x, y = stack.pop()
            for dx, dy in self.directions:
                new_x, new_y = x + dx, y + dy
                if (self.map.xy_valid(new_x, new_y) and (new_x, new_y) not in visited and
                        tiles[self.map.xy_to_i(new_x, new_y)] == '.'):
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    count += 1
        return count, visited

    def _is_connected(self):
        tiles = self.map.tiles.copy()
        start_x, start_y = None, None
        half_width = self.map.width // 2
        for y in range(1, self.map.height - 1):
            for x in range(1, half_width + 1):
                if tiles[self.map.xy_to_i(x, y)] == '.':
                    start_x, start_y = x, y
                    break
            if start_x is not None:
                break
        if start_x is None:
            return True
        reachable_count, _ = self._flood_fill(start_x, start_y, tiles)
        total_dots = sum(1 for y in range(1, self.map.height - 1) for x in range(1, half_width + 1)
                         if tiles[self.map.xy_to_i(x, y)] == '.')
        return reachable_count == total_dots

    def _is_intersection(self, x, y):
        open_paths = sum(1 for dx, dy in self.directions if self.map.get_tile(x + dx, y + dy) == '.')
        return open_paths >= 3

    def _check_path_thickness(self, x, y):
        if self.map.get_tile(x, y) != '.':
            return True
        adjacent_paths = sum(1 for dx, dy in [(1, 0), (-1, 0)] if self.map.get_tile(x + dx, y + dy) == '.')
        if adjacent_paths > 1:
            return False
        adjacent_paths = sum(1 for dx, dy in [(0, 1), (0, -1)] if self.map.get_tile(x + dx, y + dy) == '.')
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
            if dist > 0 and self._is_intersection(cx, cy) and dist < self.config["min_intersection_distance"]:
                return False
            if dist >= self.config["min_intersection_distance"]:
                continue
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if self.map.xy_valid(nx, ny) and (nx, ny) not in visited and self.map.get_tile(nx, ny) == '.':
                    queue.append((nx, ny, dist + 1))
        return True

    def _check_no_dead_ends(self):
        for y in range(1, self.map.height - 1):
            for x in range(1, self.map.width - 1):
                if self.map.get_tile(x, y) == '.':
                    open_paths = sum(1 for dx, dy in self.directions if self.map.get_tile(x + dx, y + dy) in ['.', 'T'])
                    if open_paths < 2:
                        return False
        return True

    def _can_place_shape(self, shape, center_x, center_y):
        """檢查是否可以在指定位置放置牆壁形狀。"""
        coords = self.wall_shapes[shape]
        for dx, dy in coords:
            x, y = center_x + dx, center_y + dy
            if not self.map.xy_valid(x, y) or self.map.get_tile(x, y) == '#':
                return False
        return True

    def _place_shape(self, shape, center_x, center_y):
        """放置牆壁形狀，返回影響的格子列表。"""
        coords = self.wall_shapes[shape]
        affected = []
        for dx, dy in coords:
            x, y = center_x + dx, center_y + dy
            affected.append((x, y, self.map.get_tile(x, y)))
            self.map.set_tile(x, y, 'X')
        return affected

    def _undo_shape(self, affected):
        """撤銷牆壁形狀的放置。"""
        for x, y, original in affected:
            self.map.set_tile(x, y, original)

    def _add_tunnels(self):
        half_width = self.map.width // 2
        possible_positions = []
        for x in range(2, half_width - 1):
            if self.map.get_tile(x, 1) == '.':
                possible_positions.append((x, 0))
            if self.map.get_tile(x, self.map.height - 2) == '.':
                possible_positions.append((x, self.map.height - 1))
        num_tunnels = min(self.config["num_tunnels"], len(possible_positions))
        if num_tunnels == 0:
            return
        tunnel_positions = random.sample(possible_positions, num_tunnels)
        for tx, ty in tunnel_positions:
            self.map.set_tile(tx, ty, 'T')
            self.map.set_tile(self.map.width - 1 - tx, ty, 'T')

    def _add_central_room(self):
        room = self.config["central_room"]["template"]
        room_h = len(room)
        room_w = len(room[0])
        start_x = (self.map.width - room_w) // 2
        start_y = (self.map.height - room_h) // 2

        if (start_x < 1 or start_x + room_w > self.map.width - 1 or
                start_y < 1 or start_y + room_h > self.map.height - 1):
            print("錯誤：迷宮尺寸過小，無法插入中央房間")
            sys.exit(1)

        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.map.set_tile(start_x + i, start_y + j, cell)

        entrances = [
            (start_x + 3, start_y - 1),  # 頂部
            (start_x + 3, start_y + room_h),  # 底部
            (start_x - 1, start_y + 2),  # 左側
            (start_x + room_w, start_y + 2)  # 右側
        ]
        for ex, ey in entrances:
            if self.map.xy_valid(ex, ey):
                self.map.set_tile(ex, ey, '.')
                for dx, dy in self.directions:
                    nx, ny = ex + dx, ey + dy
                    if self.map.xy_valid(nx, ny) and self.map.get_tile(nx, ny) == '.' and not self._check_path_thickness(nx, ny):
                        self.map.set_tile(nx, ny, 'X')

        while not self._is_connected():
            visited = set()
            regions = []
            half_width = self.map.width // 2
            for y in range(1, self.map.height - 1):
                for x in range(1, half_width + 1):
                    if (x, y) not in visited and self.map.get_tile(x, y) == '.':
                        _, region = self._flood_fill(x, y, self.map.tiles)
                        regions.append(region)
                        visited.update(region)

            if len(regions) <= 1:
                break

            min_dist = float('inf')
            best_pair = None
            for i, region1 in enumerate(regions):
                for j, region2 in enumerate(regions[i+1:], i+1):
                    for p1 in region1:
                        for p2 in region2:
                            dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                            if dist < min_dist:
                                min_dist = dist
                                best_pair = (p1, p2)

            if best_pair is None:
                break

            (x1, y1), (x2, y2) = best_pair
            if x1 == x2:
                y_start, y_end = min(y1, y2), max(y1, y2)
                for y in range(y_start + 1, y_end):
                    if (start_x <= x1 < start_x + room_w and start_y <= y < start_y + room_h):
                        continue
                    self.map.set_tile(x1, y, '.')
                    if not self._check_path_thickness(x1, y) or not self._check_intersection_distance(x1, y):
                        self.map.set_tile(x1, y, 'X')
                        break
            else:
                x_start, x_end = min(x1, x2), max(x1, x2)
                for x in range(x_start + 1, x_end):
                    if x > half_width:
                        break
                    if (start_x <= x < start_x + room_w and start_y <= y1 < start_y + room_h):
                        continue
                    self.map.set_tile(x, y1, '.')
                    if not self._check_path_thickness(x, y1) or not self._check_intersection_distance(x, y1):
                        self.map.set_tile(x, y1, 'X')
                        break

    def _remove_dead_ends(self):
        while True:
            modified = False
            for y in range(1, self.map.height - 1):
                for x in range(1, self.map.width - 1):
                    if self.map.get_tile(x, y) == '.':
                        open_paths = sum(1 for dx, dy in self.directions if self.map.get_tile(x + dx, y + dy) in ['.', 'T'])
                        if open_paths < 2:
                            self.map.set_tile(x, y, 'X')
                            modified = True
            if not modified:
                break

    def generate_maze(self):
        """通過放置預設牆壁形狀生成迷宮。"""
        half_width = self.map.width // 2
        shapes = list(self.wall_shapes.keys())
        target_walls = int(half_width * (self.map.height - 2) * self.config["wall_density"])
        current_walls = sum(1 for y in range(1, self.map.height - 1) for x in range(1, half_width + 1)
                            if self.map.get_tile(x, y) == 'X')
        attempts = 0
        max_attempts = 1000

        while current_walls < target_walls and attempts < max_attempts:
            x = random.randint(1, half_width)
            y = random.randint(1, self.map.height - 2)
            shape = random.choice(shapes)
            if self._can_place_shape(shape, x, y):
                affected = self._place_shape(shape, x, y)
                valid = True
                for ax, ay, _ in affected:
                    if not self._check_path_thickness(ax, ay) or not self._check_intersection_distance(ax, ay):
                        valid = False
                        break
                if valid and self._is_connected():
                    current_walls = sum(1 for y in range(1, self.map.height - 1) for x in range(1, half_width + 1)
                                        if self.map.get_tile(x, y) == 'X')
                else:
                    self._undo_shape(affected)
            attempts += 1

        self._add_central_room()
        self._add_tunnels()
        self._remove_dead_ends()

        # 鏡像到右半部分
        for y in range(self.map.height):
            for x in range(1, half_width):
                self.map.set_tile(self.map.width - 1 - x, y, self.map.get_tile(x, y))
            if self.map.width % 2 == 1 and self.map.get_tile(half_width, y) in ['A', 'S']:
                self.map.set_tile(half_width, y, '.')

def parse_arguments():
    parser = argparse.ArgumentParser(description='生成一個連通的Pac-Man迷宮')
    parser.add_argument('--width', type=int, default=29, help='迷宮寬度')
    parser.add_argument('--height', type=int, default=31, help='迷宮高度')
    parser.add_argument('--seed', type=int, help='隨機種子碼，用於可重現的生成')
    parser.add_argument('--density', type=float, default=0.4, help='牆壁密度（0.0到1.0，越大牆壁越多）')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.width < 9 or args.height < 9:
        print("錯誤：迷宮最小尺寸為 9x9 以容納中央房間")
        sys.exit(1)
    if not 0.0 <= args.density <= 1.0:
        print("錯誤：牆壁密度必須在 0.0 到 1.0 之間")
        sys.exit(1)

    base_map = BaseMap(args.width, args.height, seed=args.seed)
    generator = MazeGenerator(base_map, None)
    generator.generate_maze()

    print(f"生成的Pac-Man迷宮（種子碼：{args.seed if args.seed is not None else '無'}）")
    print(f"尺寸：{args.width}x{args.height}，牆壁密度：{args.density}")
    print(base_map)