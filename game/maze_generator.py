#!/usr/bin/python3

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

            # 如果找到交叉口，返回距離
            if self._is_intersection(cx, cy):
                return dist

            # 向所有方向擴展
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if self.xy_valid(nx, ny) and (nx, ny) not in visited and self.get_tile(nx, ny) == '.':
                    queue.append((nx, ny, dist + 1))
        return float('inf')  # 如果找不到交叉口，返回無窮大
    
    def _break_long_paths(self, max_length=4):
        """
        找到非交叉口的長路段，並打通牆壁以連接到出口或其他路徑。
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上、下、左、右
        half_width = self.w // 2  # 只處理左半邊迷宮

        for y in range(1, self.h - 1):  # 遍歷迷宮的行
            for x in range(1, half_width):  # 遍歷迷宮的列
                # 如果是空地且不是交叉口
                if self.get_tile(x, y) == '.' and not self._is_intersection(x, y):
                    # 計算與最近交叉口的距離
                    distance = self._find_nearest_intersection_distance(x, y)

                    # 如果距離超過 max_distance，隨機打通牆壁直到連接到其他路
                    if distance > max_length:
                        wall = [(x + dx, y + dy) for dx, dy in directions if self.get_tile(x + dx, y + dy) == 'X']
                        if wall:
                            # 隨機選擇一個牆壁打通
                            wall_x, wall_y = random.choice(wall)
                            self.set_tile(wall_x, wall_y, '.')
                            
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
        """
        檢測大範圍空白區域，並在中心填充牆壁。
        """
        visited = set()

        for y in range(1, self.h - 1):
            for x in range(1, self.w - 1):
                if (x, y) not in visited and self.get_tile(x, y) == '.':
                    area = self._flood_fill_get_area(x, y)
                    visited.update(area)

                    # 如果區域大小超過 max_area，則在中心填充牆壁
                    if len(area) > max_area:
                        # 計算區域的中心
                        avg_x = sum(p[0] for p in area) // len(area)
                        avg_y = sum(p[1] for p in area) // len(area)
                        self.set_tile(avg_x, avg_y, 'X')
                       
    def _add_central_room(self):
        """在迷宮中央添加一個固定的房間"""
        room = [
            "........",
            ".XX..XX.",
            ".X....X.",
            ".XXXXXX.",
            "........"
        ]
        room_h = len(room)
        room_w = len(room[0])
        start_x = (self.w - room_w) // 2
        start_y = (self.h - room_h) // 2
        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)

    def generate_connected_maze(self):
        half_width = self.w // 2
        for y in range(1, self.h - 1):
            for x in range(1, half_width):
                self.set_tile(x, y, 'X')

        start_x = random.randint(1, half_width - 1)
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
                if (1 <= new_x < half_width and 1 <= new_y < self.h - 1 and
                        self.get_tile(new_x, new_y) == 'X' and (new_x, new_y) not in visited):
                    self.set_tile(new_x, new_y, '.')
                    self.set_tile(x + dx // 2, y + dy // 2, '.')
                    visited.add((new_x, new_y))
                    stack.append((new_x, new_y))
                    found = True
                    break
            if not found:
                stack.pop()
        self._break_long_paths(max_length=4)
        self._add_central_room()
        self._fill_large_empty_areas(max_area=8)
        # 鏡像到右半部分
        for y in range(self.h):
            for x in range(half_width):
                self.set_tile(self.w - 1 - x, y, self.get_tile(x, y))


def parse_arguments():
    parser = argparse.ArgumentParser(description='生成一個連通的Pac-Man迷宮')
    parser.add_argument('--width', type=int, default=28, help='迷宮寬度（必須為偶數）')
    parser.add_argument('--height', type=int, default=31, help='迷宮高度')
    parser.add_argument('--seed', type=int, help='隨機種子碼，用於可重現的生成')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.width % 2 != 0:
        print("錯誤：迷宮寬度必須為偶數以確保對稱")
        sys.exit(1)
    if args.width < 8 or args.height < 8:
        print("錯誤：迷宮最小尺寸為 8x8")
        sys.exit(1)

    tileMap = Map(args.width, args.height, seed=args.seed)
    tileMap.generate_connected_maze()

    print(f"生成的Pac-Man迷宮（種子碼：{args.seed if args.seed is not None else '無'}）")
    print(f"尺寸：{args.width}x{args.height}")
    print(tileMap)