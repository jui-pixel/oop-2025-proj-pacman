#!/usr/bin/python3

"""
生成一個隨機的Pac-Man迷宮，確保所有空地（'.'）之間必然有通路，支持種子碼隨機化、可自定義大小和對稱佈局。
新增功能：
1. 檢測交叉口（上下左右有三個或以上的路）。
2. 找到過長的路，打通其中一面牆，連接到其他路徑。
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
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if self.xy_valid(nx, ny) and self.get_tile(nx, ny) == 'X':
                                self.set_tile(nx, ny, '.')  # 打通牆壁
                                break
                            
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

    def generate_connected_maze(self, path_density=0.5):
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
    
        # 鏡像到右半部分
        for y in range(self.h):
            for x in range(half_width):
                self.set_tile(self.w - 1 - x, y, self.get_tile(x, y))


def parse_arguments():
    parser = argparse.ArgumentParser(description='生成一個連通的Pac-Man迷宮')
    parser.add_argument('--width', type=int, default=28, help='迷宮寬度（必須為偶數）')
    parser.add_argument('--height', type=int, default=31, help='迷宮高度')
    parser.add_argument('--seed', type=int, help='隨機種子碼，用於可重現的生成')
    parser.add_argument('--density', type=float, default=0.5, help='路徑密度（0.0到1.0，越大路徑越密集）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.width % 2 != 0:
        print("錯誤：迷宮寬度必須為偶數以確保對稱")
        sys.exit(1)
    if args.width < 8 or args.height < 8:
        print("錯誤：迷宮最小尺寸為 8x8")
        sys.exit(1)
    if not 0.0 <= args.density <= 1.0:
        print("錯誤：路徑密度必須在0.0到1.0之間")
        sys.exit(1)

    tileMap = Map(args.width, args.height, seed=args.seed)
    tileMap.generate_connected_maze(path_density=args.density)

    print(f"生成的Pac-Man迷宮（種子碼：{args.seed if args.seed is not None else '無'}）")
    print(f"尺寸：{args.width}x{args.height}，路徑密度：{args.density}")
    print(tileMap)