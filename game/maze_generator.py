#!/usr/bin/python3
"""
生成一個隨機的Pac-Man迷宮，根據輸入的寬度和高度先生成矩陣，先放置中央房間，
然後在「除了該格之外的九宮格內全是路徑」的格子上加入牆壁，並以概率擴展牆壁。
滿足以下限制：
- 若新牆壁的九宮格內有不連通的牆壁（形成死路），則取消生成。
- 若新牆壁導致連通牆壁組面積超過 6 格，則取消生成。
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
            
    def _flood_fill(self, start_x, start_y, tile_type):
        """洪水填充，計算連通區域的大小和格子。"""
        if self.get_tile(start_x, start_y) != tile_type:
            return 0, set()
        stack = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        count = 1
        while stack:
            x, y = stack.pop()
            for dx, dy in self.directions:
                new_x, new_y = x + dx, y + dy
                if self.xy_valid(new_x, new_y) and (new_x, new_y) not in visited and self.get_tile(new_x, new_y) == tile_type:
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    count += 1
        return count, visited

    def _check_surrounding_paths(self, x, y):
        """檢查 (x, y) 的九宮格內除了自己外的 8 格是否全為路徑。"""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) != '.':
                    return False
        return True

    def _check_dead_end_in_neighborhood(self, x, y):
        """檢查 (x, y) 的九宮格內是否有不連通的牆壁（可能導致死路）。"""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny):
                    if self.get_tile(nx, ny) != '.' or self.get_tile(nx, ny) != 'T':
                        return False
        return True

    def _get_connected_wall_size(self, x, y):
        """計算 (x, y) 所在連通牆壁組的大小。"""
        size, _ = self._flood_fill(x, y, 'X')
        return size

    def add_initial_walls(self):
        """在九宮格內除了自己外全是路徑的格子上加入牆壁。"""
        half_width = self.width // 2
        for y in range(1, self.height - 1):
            for x in range(1, half_width + 1):
                if self.get_tile(x, y) == '.' and self._check_surrounding_paths(x, y):
                    self.set_tile(x, y, 'X')

    def extend_walls(self, extend_prob=0.5):
        """以概率在現有牆壁的上下左右生成新牆壁。"""
        half_width = self.width // 2
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            # 收集所有現有牆壁格
            wall_positions = [(x, y) for y in range(1, self.height - 1) for x in range(1, half_width + 1)
                              if self.get_tile(x, y) == 'X']
            if not wall_positions:
                break

            # 隨機選擇一個牆壁格進行擴展
            x, y = random.choice(wall_positions)
            direction = random.choice(self.directions)
            new_x, new_y = x + direction[0], y + direction[1]

            # 檢查新位置是否有效
            if (not self.xy_valid(new_x, new_y) or
                new_x > half_width or
                self.get_tile(new_x, new_y) != '.' or
                random.random() > extend_prob):
                attempts += 1
                continue

            # 臨時放置新牆壁
            self.set_tile(new_x, new_y, 'X')

            # 檢查條件 1：九宮格內是否有不連通的牆壁（可能導致死路）
            if self._check_dead_end_in_neighborhood(new_x, new_y):
                self.set_tile(new_x, new_y, '.')  # 取消生成
                attempts += 1
                continue

            # 檢查條件 2：連通牆壁組面積是否超過 6 格
            connected_size = self._get_connected_wall_size(new_x, new_y)
            if connected_size > 6:
                self.set_tile(new_x, new_y, '.')  # 取消生成
                attempts += 1
                continue

            # 檢查連通性
            if not self._is_connected():
                self.set_tile(new_x, new_y, '.')  # 取消生成
                attempts += 1
                continue

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
        """生成迷宮：先放置初始牆壁，再擴展牆壁，最後添加隧道和移除死路。"""
        self.add_initial_walls()
        self.extend_walls()
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