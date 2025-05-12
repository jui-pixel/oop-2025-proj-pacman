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
        # 如果提供了種子碼，設置隨機數生成器的種子以確保可重現的隨機結果
        if seed is not None:
            random.seed(seed)
            
        if tile_str is None:
            # 創建一個空的迷宮，初始化為指定寬高，所有格子默認為障礙物（'X'）
            self.w = w  # 迷宮寬度
            self.h = h  # 迷宮高度
            self.tiles = ['X' for _ in range(w * h)]  # 初始化所有格子為障礙物
            self._create_border()  # 自動生成迷宮邊框
        else:
            # 如果提供了地圖字符串，則根據字符串初始化迷宮
            self.setMap(w, h, tile_str)

        # 設置是否輸出詳細日誌（默認為關閉）
        self.verbose = False

    def _create_border(self):
        """創建迷宮四周的邊框障礙物"""
        # 在頂部和底部邊框設置障礙物（'#'）
        for x in range(self.w):
            self.set_tile(x, 0, '#')  # 頂部邊框
            self.set_tile(x, self.h-1, '#')  # 底部邊框
        # 在左側和右側邊框設置障礙物（'#'）
        for y in range(self.h):
            self.set_tile(0, y, '#')  # 左側邊框
            self.set_tile(self.w-1, y, '#')  # 右側邊框

    def setMap(self, w, h, tile_str):
        """根據給定的地圖字符串設置迷宮"""
        self.w = w  # 設置迷宮寬度
        self.h = h  # 設置迷宮高度
        self.tiles = list(self.format_map_str(tile_str, ""))  # 將字符串轉換為格子列表

    @staticmethod
    def format_map_str(tiles, sep):
        """格式化地圖字符串，去除多餘的換行和空格"""
        return sep.join(line.strip() for line in tiles.splitlines())

    def __str__(self):
        """將當前迷宮轉換為字符串以便顯示"""
        s = ""
        for y in range(self.h):
            for x in range(self.w):
                s += self.tiles[self.xy_to_i(x, y)]  # 按行添加每個格子的內容
            s += "\n"  # 每行結束後換行
        return s

    def xy_to_i(self, x, y):
        """將二維坐標 (x, y) 轉換為一維索引"""
        return x + y * self.w

    def i_to_xy(self, i):
        """將一維索引轉換為二維坐標 (x, y)"""
        return i % self.w, i // self.w

    def xy_valid(self, x, y):
        """檢查坐標 (x, y) 是否在迷宮範圍內"""
        return 0 <= x < self.w and 0 <= y < self.h

    def get_tile(self, x, y):
        """獲取指定坐標 (x, y) 的格子內容，若坐標無效則返回 None"""
        if not self.xy_valid(x, y):
            return None
        return self.tiles[self.xy_to_i(x, y)]

    def set_tile(self, x, y, value):
        """在指定坐標 (x, y) 設置格子內容"""
        if self.xy_valid(x, y):
            self.tiles[self.xy_to_i(x, y)] = value

    def _flood_fill(self, start_x, start_y, tiles):
        """使用洪水填充算法檢查連通性，返回可達的格子數"""
        if tiles[self.xy_to_i(start_x, start_y)] != '.':
            return 0
        stack = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        count = 1
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右
        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (self.xy_valid(new_x, new_y) and (new_x, new_y) not in visited and
                        tiles[self.xy_to_i(new_x, new_y)] == '.'):
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    count += 1
        return count

    def _is_connected(self):
        """檢查迷宮中所有空地是否連通"""
        # 複製當前迷宮狀態
        tiles = self.tiles.copy()
        # 找到第一個空地作為起點
        start_x, start_y = None, None
        for y in range(1, self.h - 1):
            for x in range(1, self.w // 2):
                if tiles[self.xy_to_i(x, y)] == '.':
                    start_x, start_y = x, y
                    break
            if start_x is not None:
                break
        if start_x is None:
            return True  # 沒有空地，認為連通
        # 使用洪水填充計算可達格子數
        reachable_count = self._flood_fill(start_x, start_y, tiles)
        # 統計左半部分的所有空地數
        total_dots = sum(1 for y in range(1, self.h - 1) for x in range(1, self.w // 2)
                         if tiles[self.xy_to_i(x, y)] == '.')
        return reachable_count == total_dots

    def generate_connected_maze(self, path_density=0.7):
        """使用改進的DFS生成連通的迷宮，確保所有空地之間有通路，並保持對稱"""
        half_width = self.w // 2  # 只在左半部分生成迷宮
        # 初始化迷宮，所有非邊框格子為障礙物（'X'）
        for y in range(1, self.h - 1):
            for x in range(1, half_width):
                self.set_tile(x, y, 'X')

        # 隨機選擇起點（避開邊框）
        start_x = random.randint(1, half_width - 1)
        start_y = random.randint(1, self.h - 2)
        self.set_tile(start_x, start_y, '.')  # 設置起點為空地

        # 使用DFS生成連通路徑
        stack = [(start_x, start_y)]
        visited = {(start_x, start_y)}
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]  # 上下左右，步長為2以留出牆壁

        while stack:
            x, y = stack[-1]
            random.shuffle(directions)  # 隨機打亂方向以增加隨機性
            found = False
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                # 檢查新位置是否有效且未訪問
                if (1 <= new_x < half_width and 1 <= new_y < self.h - 1 and
                        self.get_tile(new_x, new_y) == 'X' and (new_x, new_y) not in visited):
                    # 設置新位置為空地
                    self.set_tile(new_x, new_y, '.')
                    # 設置中間格子為空地（移除牆壁）
                    self.set_tile(x + dx // 2, y + dy // 2, '.')
                    visited.add((new_x, new_y))
                    stack.append((new_x, new_y))
                    found = True
                    break
            if not found:
                stack.pop()

        # 根據路徑密度隨機添加障礙物，確保不破壞連通性
        target_dots = int((half_width - 1) * (self.h - 2) * path_density)
        current_dots = sum(1 for y in range(1, self.h - 1) for x in range(1, half_width)
                           if self.get_tile(x, y) == '.')
        attempts = 0
        max_attempts = 1000  # 防止無限循環
        while current_dots > target_dots and attempts < max_attempts:
            x = random.randint(1, half_width - 1)
            y = random.randint(1, self.h - 2)
            if self.get_tile(x, y) == '.':
                # 模擬添加障礙物
                self.set_tile(x, y, 'X')
                if not self._is_connected():
                    # 如果破壞連通性，撤銷
                    self.set_tile(x, y, '.')
                else:
                    current_dots -= 1
                attempts += 1

        # 鏡像到右半部分
        for y in range(self.h):
            for x in range(half_width):
                self.set_tile(self.w - 1 - x, y, self.get_tile(x, y))

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='生成一個連通的Pac-Man迷宮')
    parser.add_argument('--width', type=int, default=28, help='迷宮寬度（必須為偶數）')
    parser.add_argument('--height', type=int, default=31, help='迷宮高度')
    parser.add_argument('--seed', type=int, help='隨機種子碼，用於可重現的生成')
    parser.add_argument('--density', type=float, default=0.7, help='路徑密度（0.0到1.0，越大路徑越密集）')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    # python .\game\maze_generator.py --width 28 --height 31 --seed 123 --density 0.8
    # 驗證迷宮尺寸
    if args.width % 2 != 0:
        print("錯誤：迷宮寬度必須為偶數以確保對稱")
        sys.exit(1)
    if args.width < 8 or args.height < 8:
        print("錯誤：迷宮最小尺寸為 8x8")
        sys.exit(1)
    if not 0.0 <= args.density <= 1.0:
        print("錯誤：路徑密度必須在0.0到1.0之間")
        sys.exit(1)

    # 使用指定的種子碼創建迷宮
    tileMap = Map(args.width, args.height, seed=args.seed)

    # 生成連通的迷宮
    tileMap.generate_connected_maze(path_density=args.density)

    # 輸出生成的迷宮
    print(f"生成的Pac-Man迷宮（種子碼：{args.seed if args.seed is not None else '無'}）")
    print(f"尺寸：{args.width}x{args.height}，路徑密度：{args.density}")
    print(tileMap)