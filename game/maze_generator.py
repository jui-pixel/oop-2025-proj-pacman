#game/maze_generator.py
#!/usr/bin/python3
"""
# : 邊界
. : 空地
X : 牆壁
D : 牆壁門口
T : 暫時的牆壁(方便生成)
"""

import sys
import random

class Map:
    def __init__(self, w, h, seed=None):
        if seed is not None:
            random.seed(seed)
        self.width = w
        self.height = h
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
            ".XXDXX.",
            ".XSSSX.",
            ".XXXXX.",
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
                    if self.get_tile(nx, ny) not in ['.', 'T']:
                        return False
        return True

    def _get_connected_wall_size(self, x, y):
        """計算 (x, y) 所在連通牆壁組的大小。"""
        size, _ = self._flood_fill(x, y, 'T')
        return size

    def valid_wall_spawnpoint(self, x, y):
        if self.xy_valid(x, y):
            if self._check_surrounding_paths(x, y) and self.get_tile(x, y) == '.': 
                return True
        return False
    
    def convert_nearby_T_to_wall(self, x, y):
        """將 (x, y) 附近的 T 轉換為牆壁。"""
        if self.get_tile(x, y) == 'T':
            _, connected_tiles = self._flood_fill(x, y, 'T')
            for cx, cy in connected_tiles:
                self.set_tile(cx, cy, 'X')
    def convert_all_T_to_wall(self):
        """將所有 T 轉換為牆壁。"""
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == 'T':
                    self.set_tile(x, y, 'X')

    def extend_walls(self, extend_prob=0.9):
        """以概率在現有牆壁的上下左右生成新牆壁。"""
        half_width = self.width // 2
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            attempts += 1
            # 收集所有現有牆壁格
            wall_positions = [(x, y) for y in range(1, self.height - 1) for x in range(1, half_width + 1)
                              if self.valid_wall_spawnpoint(x, y)]
            if not wall_positions:
                break

            # 隨機選擇一個牆壁格進行擴展
            x, y = random.choice(wall_positions)
            self.set_tile(x, y, 'T')
            if random.random() > extend_prob:
                continue
            direction = random.choice(self.directions)
            new_x, new_y = x + direction[0], y + direction[1]
            # 檢查新位置是否有效
            while self._check_dead_end_in_neighborhood(new_x, new_y):# 檢查條件 1：九宮格內是否有不連通的牆壁（可能導致死路）
                self.set_tile(new_x, new_y, 'T')
                  
                # 檢查條件 2：連通牆壁組面積是否超過 4 格
                connected_size = self._get_connected_wall_size(new_x, new_y)
                if connected_size > 4:
                    break
                if random.random() < extend_prob:
                    direction = random.choice(self.directions)
                    new_x, new_y = new_x + direction[0], new_y + direction[1]
                else:
                    break
                
            # 將所有 T 轉換為牆壁
            self.convert_nearby_T_to_wall(x, y)
            
    def if_dead_end(self, x, y):
        """檢查 (x, y) 是否是死路。"""
        if not self.xy_valid(x, y):
            return False
        if self.get_tile(x, y) != '.':
            return False
        return sum(1 for dx, dy in self.directions if self.get_tile(x + dx, y + dy) != '.') >= 3
    
    def _check_connectivity(self, area, blocked_cell=None):
        """檢查區域內所有格子是否連通，允許臨時阻擋一個格子。"""
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
        """縮窄 2x2 空地塊，隨機選擇牆壁位置，確保不產生死路，返回放置牆壁次數"""
        count = 0
        S = 'A'  # 臨時牆壁標記
        for y in range(1, self.height - 2):
            for x in range(1, self.width - 2):
                # 檢查是否為 2x2 空地塊
                if not (self.get_tile(x, y) == '.' and self.get_tile(x + 1, y) == '.' and
                        self.get_tile(x, y + 1) == '.' and self.get_tile(x + 1, y + 1) == '.'):
                    continue

                # 收集 2x2 塊內的格子
                block = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                # 收集周圍的空地鄰居（用於連通性檢查）
                neighbors = set()
                for bx, by in block:
                    for dx, dy in self.directions:
                        nx, ny = bx + dx, by + dy
                        if self.xy_valid(nx, ny) and self.get_tile(nx, ny) == '.' and (nx, ny) not in block:
                            neighbors.add((nx, ny))

                placed = False
                # 定義四個角落條件
                conditions = [
                    ((x, y), lambda: self.get_tile(x - 1, y) != '.' and self.get_tile(x, y - 1) != '.'),
                    ((x + 1, y), lambda: self.get_tile(x + 2, y) != '.' and self.get_tile(x + 1, y - 1) != '.'),
                    ((x, y + 1), lambda: self.get_tile(x, y + 2) != '.' and self.get_tile(x - 1, y + 1) != '.'),
                    ((x + 1, y + 1), lambda: self.get_tile(x + 2, y + 1) != '.' and self.get_tile(x + 1, y + 2) != '.')
                ]
                # 隨機化角落檢查順序
                random.shuffle(conditions)

                # 嘗試角落條件
                for (cx, cy), condition in conditions:
                    if condition() and self._check_connectivity(neighbors, (cx, cy)):
                        self.set_tile(cx, cy, S)
                        # 檢查周圍是否產生死路
                        dead_end = False
                        for dx, dy in self.directions:
                            nx, ny = cx + dx, cy + dy
                            if self.if_dead_end(nx, ny):
                                dead_end = True
                                break
                        if not dead_end:
                            placed = True
                            count += 1
                            break
                        self.set_tile(cx, cy, '.')  # 恢復

                # 如果角落條件不滿足，隨機嘗試 2x2 塊內的格子
                if not placed:
                    random.shuffle(block)
                    for bx, by in block:
                        if self._check_connectivity(neighbors, (bx, by)):
                            self.set_tile(bx, by, S)
                            dead_end = False
                            for dx, dy in self.directions:
                                nx, ny = bx + dx, by + dy
                                if self.if_dead_end(nx, ny):
                                    dead_end = True
                                    break
                            if not dead_end:
                                placed = True
                                count += 1
                                break
                            self.set_tile(bx, by, '.')  # 恢復

        # 將所有臨時標記 'A' 轉為牆壁 'X'
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == 'A':
                    self.set_tile(x, y, 'X')

        return count
                
                    
        
    def generate_maze(self):
        """生成迷宮：先放置初始牆壁，再擴展牆壁，最後添加隧道和移除死路。"""
        
        self.extend_walls()
        self.convert_all_T_to_wall()
        
        # 鏡像到右半部分
        half_width = self.width // 2
        for y in range(self.height):
            for x in range(1, half_width):
                self.set_tile(self.width - 1 - x, y, self.get_tile(x, y))
        while self.narrow_paths():
            pass

if __name__ == "__main__":
    width = 25
    height = 25
    seed = 1

    if width < 9 or height < 9:
        print("錯誤：迷宮最小尺寸為 9x9 以容納中央房間")
        sys.exit(1)

    maze = Map(width, height, seed=seed)
    maze.generate_maze()

    print(f"生成的Pac-Man迷宮（種子碼：{seed if seed is not None else '無'}）")
    print(f"尺寸：{width}x{height}")
    print(maze)