#!/usr/bin/python3

"""
生成一個隨機的Pac-Man迷宮，確保所有空地（'.'）之間必然有通路，支持種子碼隨機化、可自定義大小和對稱佈局。
所有障礙物統一使用代號 'X'，死路及偏僻區域標記為 'A'，幽靈生成位置標記為 'S'。
包含打破長路、填充大空地、強制插入7x5中央房間等功能，支持奇數寬度。
"""

import sys
import random
import argparse
from copy import deepcopy

class Map:
    def __init__(self, w, h, seed=None, tile_str=None):
        if seed is not None:
            random.seed(seed)

        if tile_str is None:
            self.w = w
            self.h = h
            self.tiles = ['X' for _ in range(w * h)]
            self._create_border()
        else:
            self.setMap(w, h, tile_str)

    def _create_border(self):
        for x in range(self.w):
            self.set_tile(x, 0, '#')
            self.set_tile(x, self.h-1, '#')
        for y in range(self.h):
            self.set_tile(0, y, '#')
            self.set_tile(self.w-1, y, '#')

    def setMap(self, w, h, tile_str):
        self.w = w
        self.h = h
        self.tiles = list(self.format_map_str(tile_str, ""))

    @staticmethod
    def format_map_str(tiles, sep):
        return sep.join(line.strip() for line in tiles.splitlines())

    def __str__(self):
        s = ""
        for y in range(self.h):
            for x in range(self.w):
                s += self.tiles[self.xy_to_i(x, y)]
            s += "\n"
        return s

    def xy_to_i(self, x, y):
        return x + y * self.w

    def i_to_xy(self, i):
        return i % self.w, i // self.w

    def xy_valid(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def get_tile(self, x, y):
        if not self.xy_valid(x, y):
            return None
        return self.tiles[self.xy_to_i(x, y)]

    def set_tile(self, x, y, value):
        if self.xy_valid(x, y):
            self.tiles[self.xy_to_i(x, y)] = value

    def _flood_fill(self, start_x, start_y, tiles):
        """使用洪水填充算法檢查連通性，返回可達的格子數和訪問的格子集合"""
        if tiles[self.xy_to_i(start_x, start_y)] != '.':
            return 0, set()
        stack = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        count = 1
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (self.xy_valid(new_x, new_y) and (new_x, new_y) not in visited and
                        tiles[self.xy_to_i(new_x, new_y)] == '.'):
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    count += 1
        return count, visited

    def _is_connected(self):
        """檢查迷宮中所有空地（'.'）是否連通"""
        tiles = self.tiles.copy()
        start_x, start_y = None, None
        half_width = self.w // 2
        for y in range(1, self.h - 1):
            for x in range(1, half_width + 1):
                if tiles[self.xy_to_i(x, y)] == '.':
                    start_x, start_y = x, y
                    break
            if start_x is not None:
                break
        if start_x is None:
            return True  # 沒有空地，認為連通
        reachable_count, _ = self._flood_fill(start_x, start_y, tiles)
        total_dots = sum(1 for y in range(1, self.h - 1) for x in range(1, half_width + 1)
                         if tiles[self.xy_to_i(x, y)] == '.')
        return reachable_count == total_dots

    def _is_intersection(self, x, y):
        """檢測某個位置是否為交叉口（上下左右有三個或以上的方向是空地）"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        open_paths = 0
        for dx, dy in directions:
            if self.get_tile(x + dx, y + dy) == '.':
                open_paths += 1
        return open_paths >= 3

    def _find_nearest_intersection_distance(self, x, y):
        """計算當前位置與最近交叉口的距離"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        visited = set()
        queue = [(x, y, 0)]  # (x位置, y位置, 距離)
        while queue:
            cx, cy, dist = queue.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            if self._is_intersection(cx, cy):
                return dist
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if self.xy_valid(nx, ny) and (nx, ny) not in visited and self.get_tile(nx, ny) == '.':
                    queue.append((nx, ny, dist + 1))
        return float('inf')

    def _break_long_paths(self, max_length=4):
        """找到非交叉口的長路段，並打通牆壁以連接到出口或其他路徑"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        half_width = self.w // 2
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            modified = False
            for y in range(1, self.h - 1):
                for x in range(1, half_width + 1):
                    if self.get_tile(x, y) == '.' and not self._is_intersection(x, y):
                        distance = self._find_nearest_intersection_distance(x, y)
                        if distance > max_length:
                            wall = [(x + dx, y + dy) for dx, dy in directions if self.get_tile(x + dx, y + dy) == 'X']
                            if wall:
                                wall_x, wall_y = random.choice(wall)
                                original = self.get_tile(wall_x, wall_y)
                                self.set_tile(wall_x, wall_y, '.')
                                if not self._is_connected():
                                    self.set_tile(wall_x, wall_y, original)
                                else:
                                    modified = True
            if not modified:
                break
            attempts += 1

    def _flood_fill_get_area(self, x, y):
        """使用洪水填充找到連通的空白區域"""
        if self.get_tile(x, y) != '.':
            return []
        stack = [(x, y)]
        visited = set()
        area = []
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            area.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) == '.' and (nx, ny) not in visited:
                    stack.append((nx, ny))
        return area

    def _fill_large_empty_areas(self, max_area=8):
        """檢測大範圍空白區域，並在中心填充牆壁"""
        visited = set()
        half_width = self.w // 2
        for y in range(1, self.h - 1):
            for x in range(1, half_width + 1):
                if (x, y) not in visited and self.get_tile(x, y) == '.':
                    area = self._flood_fill_get_area(x, y)
                    visited.update(area)
                    if len(area) > max_area:
                        avg_x = sum(p[0] for p in area) // len(area)
                        avg_y = sum(p[1] for p in area) // len(area)
                        original = self.get_tile(avg_x, avg_y)
                        self.set_tile(avg_x, avg_y, 'X')
                        if not self._is_connected():
                            self.set_tile(avg_x, avg_y, original)

    def _mark_dead_ends_and_isolated_areas(self, max_distance=7):
        """在死路及偏僻的地方做標記 'A'，避免中央列"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        half_width = self.w // 2
        for y in range(1, self.h - 1):
            for x in range(1, half_width):
                if self.get_tile(x, y) == '.':
                    open_paths = sum(1 for dx, dy in directions if self.get_tile(x + dx, y + dy) == '.' or self.get_tile(x + dx, y + dy) == 'S' or self.get_tile(x + dx, y + dy) == 'A')
                    if open_paths == 1:
                        self.set_tile(x, y, 'A')
                        continue
                    distance = self._find_nearest_intersection_distance(x, y)
                    if distance > max_distance:
                        self.set_tile(x, y, 'A')

    def _add_central_room(self):
        """強制在迷宮中央添加7x5房間，包含幽靈生成位置 'S'，並確保連通性"""
        room = [
            ".......",
            ".XX.XX.",
            ".XS.SX.",
            ".XXXXX.",
            "......."
        ]
        room_h = len(room)
        room_w = len(room[0])
        start_x = (self.w - room_w) // 2
        start_y = (self.h - room_h) // 2

        # 檢查房間是否能放入迷宮
        if (start_x < 1 or start_x + room_w > self.w - 1 or start_y < 1 or start_y + room_h > self.h - 1):
            print("錯誤：迷宮尺寸過小，無法插入中央房間")
            sys.exit(1)

        # 插入房間
        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)

        # 添加初始入口
        entrances = [
            (start_x + 3, start_y - 1),  # 頂部
            (start_x + 3, start_y + room_h),  # 底部
            (start_x - 1, start_y + 2),  # 左側
            (start_x + room_w, start_y + 2)  # 右側
        ]
        for ex, ey in entrances:
            if self.xy_valid(ex, ey):
                self.set_tile(ex, ey, '.')

        # 如果不連通，找到所有連通區域並連接它們
        while not self._is_connected():
            # 找到所有連通區域
            visited = set()
            regions = []
            half_width = self.w // 2
            for y in range(1, self.h - 1):
                for x in range(1, half_width + 1):
                    if (x, y) not in visited and self.get_tile(x, y) == '.':
                        _, region = self._flood_fill(x, y, self.tiles)
                        regions.append(region)
                        visited.update(region)

            if len(regions) <= 1:
                break  # 只有一個區域，應該已經連通

            # 選擇兩個區域並找到最近的點對
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

            # 連接兩個區域（沿水平或垂直路徑）
            (x1, y1), (x2, y2) = best_pair
            if x1 == x2:  # 垂直連接
                y_start, y_end = min(y1, y2), max(y1, y2)
                for y in range(y_start + 1, y_end):
                    # 避免修改房間內的格子
                    if (start_x <= x1 < start_x + room_w and
                            start_y <= y < start_y + room_h):
                        continue
                    self.set_tile(x1, y, '.')
            else:  # 水平連接
                x_start, x_end = min(x1, x2), max(x1, x2)
                for x in range(x_start + 1, x_end):
                    if x > half_width:  # 限制在左半邊和中央列
                        break
                    if (start_x <= x < start_x + room_w and
                            start_y <= y1 < start_y + room_h):
                        continue
                    self.set_tile(x, y1, '.')

    def generate_connected_maze(self, path_density=0.7):
        half_width = self.w // 2
        for y in range(1, self.h - 1):
            for x in range(1, half_width + 1):
                self.set_tile(x, y, 'X')

        start_x = random.randint(1, half_width)
        start_y = random.randint(1, self.h - 2)
        self.set_tile(start_x, start_y, '.')

        stack = [(start_x, start_y)]
        visited = {(start_x, start_y)}
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

        while stack:
            x, y = stack[-1]
            random.shuffle(directions)
            found = False
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (1 <= new_x <= half_width and 1 <= new_y < self.h - 1 and
                        self.get_tile(new_x, new_y) == 'X' and (new_x, new_y) not in visited):
                    self.set_tile(new_x, new_y, '.')
                    self.set_tile(x + dx // 2, y + dy // 2, '.')
                    visited.add((new_x, new_y))
                    stack.append((new_x, new_y))
                    found = True
                    break
            if not found:
                stack.pop()

        # 調整路徑密度
        target_dots = int((half_width) * (self.h - 2) * path_density)
        current_dots = sum(1 for y in range(1, self.h - 1) for x in range(1, half_width + 1)
                           if self.get_tile(x, y) == '.')
        attempts = 0
        max_attempts = 1000
        while current_dots > target_dots and attempts < max_attempts:
            x = random.randint(1, half_width)
            y = random.randint(1, self.h - 2)
            if self.get_tile(x, y) == '.':
                original = self.get_tile(x, y)
                self.set_tile(x, y, 'X')
                if not self._is_connected():
                    self.set_tile(x, y, original)
                else:
                    current_dots -= 1
                attempts += 1

        self._break_long_paths(max_length=4)
        self._fill_large_empty_areas(max_area=8)
        self._add_central_room()
        self._mark_dead_ends_and_isolated_areas(max_distance=3)

        # 鏡像到右半部分，確保中央列自對稱
        for y in range(self.h):
            for x in range(1, half_width):
                self.set_tile(self.w - 1 - x, y, self.get_tile(x, y))
            if self.w % 2 == 1:
                if self.get_tile(half_width, y) in ['A', 'S']:
                    self.set_tile(half_width, y, '.')

def parse_arguments():
    parser = argparse.ArgumentParser(description='生成一個連通的Pac-Man迷宮')
    parser.add_argument('--width', type=int, default=29, help='迷宮寬度')
    parser.add_argument('--height', type=int, default=31, help='迷宮高度')
    parser.add_argument('--seed', type=int, help='隨機種子碼，用於可重現的生成')
    parser.add_argument('--density', type=float, default=0.7, help='路徑密度（0.0到1.0，越大路徑越密集）')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.width < 9 or args.height < 9:
        print("錯誤：迷宮最小尺寸為 9x9 以容納中央房間")
        sys.exit(1)
    if not 0.0 <= args.density <= 1.0:
        print("錯誤：路徑密度必須在0.0到1.0之間")
        sys.exit(1)

    tileMap = Map(args.width, args.height, seed=args.seed)
    tileMap.generate_connected_maze(path_density=args.density)

    print(f"生成的Pac-Man迷宮（種子碼：{args.seed if args.seed is not None else '無'}）")
    print(f"尺寸：{args.width}x{args.height}，路徑密度：{args.density}")
    print(tileMap)