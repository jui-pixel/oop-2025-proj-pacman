#!/usr/bin/python3

"""
生成一個隨機的Pac-Man迷宮，確保所有空地（'.'）之間必然有通路，支持種子碼隨機化、可自定義大小和對稱佈局。
所有障礙物統一使用代號 'X'。
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

    def _tighten_walls(self):
        """增加牆壁的緊湊性"""
        half_width = self.w // 2
        for y in range(1, self.h - 1):
            for x in range(1, half_width):
                # 提高牆的生成機率
                if self.get_tile(x, y) == 'X' and random.random() < 0.7:
                    self.set_tile(x, y, 'X')
                elif self.get_tile(x, y) == '.':
                    self.set_tile(x, y, '.')

    def _add_central_room(self):
        """在迷宮中央添加一個固定的房間"""
        room = [
            ".......",
            ".XX.XX.",
            ".X...X.",
            ".XXXXX.",
            "......."
        ]
        room_h = len(room)
        room_w = len(room[0])
        start_x = (self.w - room_w) // 2
        start_y = (self.h - room_h) // 2
        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)

    def _flood_fill(self, start_x, start_y):
        """從中央房間開始檢查所有可達空地"""
        visited = set()
        stack = [(start_x, start_y)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while stack:
            x, y = stack.pop()
            if (x, y) in visited or not self.xy_valid(x, y):
                continue
            if self.get_tile(x, y) != '.':
                continue
            visited.add((x, y))
            for dx, dy in directions:
                stack.append((x + dx, y + dy))
        return visited

    def _close_unreachable_paths(self):
        """將無法從中央房間到達的空地轉換為牆壁"""
        # 找到中央房間的起始點
        room_x = self.w // 2
        room_y = self.h // 2
        reachable = self._flood_fill(room_x, room_y)

        # 將無法到達的空地變為牆壁
        for y in range(1, self.h - 1):
            for x in range(1, self.w - 1):
                if (x, y) not in reachable and self.get_tile(x, y) == '.':
                    self.set_tile(x, y, 'X')
    def _break_long_paths(self, max_length=5):
        """找到過長的路，並打通其中一面牆以連接到其他路徑。"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上、下、左、右
        half_width = self.w // 2  # 只處理左半邊迷宮

        for y in range(1, self.h - 1):  # 遍歷迷宮的行
            for x in range(1, half_width):  # 遍歷迷宮的列
                # 檢測是否是過長的路
                if self.get_tile(x, y) == '.' and not self._is_intersection(x, y):
                    for dx, dy in directions:  # 對每個方向檢測路徑
                        length = 0
                        nx, ny = x + dx, y + dy
                        # 計算該方向的連續空地數量
                        while self.xy_valid(nx, ny) and self.get_tile(nx, ny) == '.':
                            length += 1
                            nx += dx
                            ny += dy
                        # 如果路徑長度超過 max_length，打通一面牆
                        if length > max_length:
                            wall_x, wall_y = x + dx * (length // 2), y + dy * (length // 2)  # 中間位置
                            for wx, wy in directions:  # 找到牆壁並打通
                                if self.get_tile(wall_x + wx, wall_y + wy) == 'X':
                                    self.set_tile(wall_x + wx, wall_y + wy, '.')  # 打通牆壁
                                    break
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

        self._tighten_walls()
        self._add_central_room()
        self._close_unreachable_paths()
    
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