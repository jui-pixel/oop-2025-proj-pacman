# game/maze_generator.py
"""
迷宮生成模組，負責生成隨機的 Pac-Man 迷宮，包含牆壁、路徑、能量球和鬼魂重生點。
使用隨機牆壁擴展和路徑縮窄演算法，確保迷宮連通且具有挑戰性。

這個模組為 Pac-Man 遊戲生成對稱的迷宮，左半部分生成後鏡像到右半部分，確保結構美觀且可玩。
"""

# 匯入必要的模組
import sys
import os
# 將父目錄添加到系統路徑，確保可以匯入其他模組（例如 config）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random  # 用於隨機生成牆壁和放置能量球
# 從 config 檔案匯入常數，例如迷宮尺寸和圖塊類型
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, TILE_BOUNDARY, TILE_WALL, TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR, TILE_TEMP_WALL, TILE_TEMP_MARKER

class Map:
    def __init__(self, width, height, seed=None):
        """
        初始化迷宮，設置尺寸和隨機種子。

        原理：
        - 創建迷宮物件，指定寬度和高度，初始化所有格子為路徑（TILE_PATH）。
        - 若提供種子，設置隨機數生成器以確保迷宮可重現。
        - 定義四個移動方向（上下左右），用於牆壁擴展和連通性檢查。

        Args:
            width (int): 迷宮寬度（格子數）。
            height (int): 迷宮高度（格子數）。
            seed (int, optional): 隨機種子，用於生成可重現的迷宮。
        """
        # 如果提供了種子，設置隨機數生成器，確保迷宮生成結果可重現
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        # 設置迷宮寬度和高度
        self.width = width
        self.height = height
        # 初始化迷宮格子陣列，所有格子初始為路徑
        self.tiles = [TILE_PATH for _ in range(self.width * self.height)]
        # 定義四個移動方向：下、上、右、左
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 初始化迷宮邊界
        self._initialize_map()

    def _initialize_map(self):
        """
        初始化迷宮邊界，設置為邊界圖塊。

        原理：
        - 將迷宮四周的格子設置為邊界（TILE_BOUNDARY），防止實體越界。
        - 邊界圖塊不可穿越，用於限制 Pac-Man 和鬼魂的移動範圍。
        """
        # 設置頂部和底部邊界
        for x in range(self.width):
            self.set_tile(x, 0, TILE_BOUNDARY)  # 頂部邊界
            self.set_tile(x, self.height - 1, TILE_BOUNDARY)  # 底部邊界
        # 設置左側和右側邊界
        for y in range(self.height):
            self.set_tile(0, y, TILE_BOUNDARY)  # 左側邊界
            self.set_tile(self.width // 2 + 1, y, TILE_BOUNDARY)  # 右側邊界（迷宮中間）

    def add_central_room(self):
        """
        在迷宮中央添加 5x5 的房間，包含鬼魂重生點和門。

        原理：
        - 在迷宮中央創建一個 5x5 的房間，包含：
          - 鬼魂重生點（TILE_GHOST_SPAWN），位於房間中心。
          - 門（TILE_DOOR），限制鬼魂進出。
          - 能量球（TILE_POWER_PELLET），位於房間四角。
          - 牆壁（TILE_WALL）和路徑（TILE_PATH），形成房間結構。
        - 房間位置計算公式：start_x = (width - 5) // 2, start_y = (height - 5) // 2。
        - 若迷宮尺寸過小（無法容納房間），則報錯並退出。

        Raises:
            SystemExit: 若迷宮尺寸不足以容納中央房間。
        """
        # 定義 5x5 房間結構，每行是一個字串，表示不同圖塊
        room = [
            f"{TILE_PATH}{TILE_PATH}{TILE_PATH}{TILE_PATH}{TILE_PATH}",  # 第一行：全為路徑
            f"{TILE_PATH}{TILE_WALL}{TILE_DOOR}{TILE_WALL}{TILE_PATH}",  # 第二行：路徑-牆壁-門-牆壁-路徑
            f"{TILE_PATH}{TILE_DOOR}{TILE_GHOST_SPAWN}{TILE_DOOR}{TILE_PATH}",  # 第三行：路徑-門-重生點-門-路徑
            f"{TILE_PATH}{TILE_WALL}{TILE_DOOR}{TILE_WALL}{TILE_PATH}",  # 第四行：路徑-牆壁-門-牆壁-路徑
            f"{TILE_PATH}{TILE_PATH}{TILE_PATH}{TILE_PATH}{TILE_PATH}"  # 第五行：全為路徑
        ]
        room_w, room_h = 5, 5  # 房間尺寸
        # 計算房間左上角坐標，確保房間位於迷宮中心
        start_x = (self.width - room_w) // 2
        start_y = (self.height - room_h) // 2
        
        # 檢查房間是否能放入迷宮
        if start_x < 1 or start_x + room_w > self.width - 1 or start_y < 1 or start_y + room_h > self.height - 1:
            print("錯誤：迷宮尺寸過小，無法插入中央房間")
            sys.exit(1)
        
        # 將房間結構設置到迷宮中
        for j, row in enumerate(room):
            for i, cell in enumerate(row):
                self.set_tile(start_x + i, start_y + j, cell)

    def __str__(self):
        """
        返回迷宮的字串表示。

        原理：
        - 將迷宮的格子陣列轉換為多行字串，每行表示迷宮的一行格子。
        - 用於調試或顯示迷宮結構。

        Returns:
            str: 迷宮的字串表示，每個格子由對應的圖塊字符表示，行間以換行符分隔。
        """
        s = ""
        for y in range(self.height):
            for x in range(self.width):
                s += self.tiles[self.xy_to_i(x, y)]  # 添加當前格子的圖塊字符
            s += "\n"  # 每行結束後換行
        return s

    def xy_to_i(self, x, y):
        """
        將 (x, y) 坐標轉換為一維索引。

        原理：
        - 迷宮格子儲存在一維陣列中，索引計算公式：i = x + y * width。
        - 用於快速訪問和修改格子。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。

        Returns:
            int: 對應的一維索引。
        """
        return x + y * self.width

    def i_to_xy(self, i):
        """
        將一維索引轉換為 (x, y) 坐標。

        原理：
        - 逆向計算坐標：x = i % width, y = i // width。
        - 用於從索引反推格子位置。

        Args:
            i (int): 一維索引。

        Returns:
            Tuple[int, int]: (x, y) 坐標。
        """
        return i % self.width, i // self.width

    def xy_valid(self, x, y):
        """
        檢查座標是否有效。

        原理：
        - 驗證坐標是否在迷宮範圍內：0 <= x < width 且 0 <= y < height。
        - 用於防止越界訪問。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。

        Returns:
            bool: 是否在迷宮範圍內。
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, x, y):
        """
        獲取指定座標的圖塊。

        原理：
        - 根據坐標計算索引，返回對應格子的圖塊類型。
        - 若坐標無效，返回 None。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。

        Returns:
            str or None: 圖塊類型（TILE_PATH、TILE_WALL 等）或 None（無效坐標）。
        """
        if not self.xy_valid(x, y):
            return None
        return self.tiles[self.xy_to_i(x, y)]

    def set_tile(self, x, y, value):
        """
        設置指定座標的圖塊類型。

        原理：
        - 根據坐標計算索引，將指定格子設置為給定的圖塊類型。
        - 僅在坐標有效時執行。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            value (str): 圖塊類型（TILE_PATH、TILE_WALL 等）。
        """
        if self.xy_valid(x, y):
            self.tiles[self.xy_to_i(x, y)] = value

    def _flood_fill(self, start_x, start_y, tile_type):
        """
        使用洪水填充演算法計算連通區域的大小和格子。

        原理：
        - 從指定起點開始，使用堆疊實現的洪水填充，遍歷所有與起點連通的同類型格子。
        - 記錄訪問的格子和連通區域大小，用於檢查迷宮連通性或牆壁組大小。
        - 每次檢查四個方向（上下左右），僅訪問未訪問且類型匹配的格子。

        Args:
            start_x (int): 起始 x 坐標。
            start_y (int): 起始 y 坐標。
            tile_type (str): 要填充的圖塊類型（例如 TILE_PATH 或 TILE_TEMP_WALL）。

        Returns:
            Tuple[int, set]: (連通區域大小, 訪問的格子集合)。
        """
        # 如果起點不是指定圖塊類型，返回空結果
        if self.get_tile(start_x, start_y) != tile_type:
            return 0, set()
        # 初始化堆疊和已訪問集合
        stack = [(start_x, start_y)]
        visited = set([(start_x, start_y)])
        count = 1  # 計數起點格子
        # 遍歷堆疊
        while stack:
            x, y = stack.pop()
            # 檢查四個方向
            for dx, dy in self.directions:
                new_x, new_y = x + dx, y + dy
                # 如果新位置有效、未訪問且為指定圖塊類型
                if (self.xy_valid(new_x, new_y) and (new_x, new_y) not in visited and 
                    self.get_tile(new_x, new_y) == tile_type):
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    count += 1
        return count, visited

    def _check_if_surrounding_all_wall_type(self, x, y, wall_type):
        """
        檢查 (x, y) 周圍九宮格是否全為指定圖塊（不包括自己）。

        原理：
        - 檢查 (x, y) 格子周圍的八個格子（3x3 區域，排除中心點），是否全部為指定圖塊（例如路徑）。
        - 用於確保牆壁生成點不會破壞迷宮結構。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            wall_type (list): 指定的圖塊類型列表。

        Returns:
            bool: 是否全為指定圖塊。
        """
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) not in wall_type:
                    return False
        return True

    def _check_if_four_edges_all_wall_type(self, x, y, wall_type):
        """
        檢查 (x, y) 四個邊緣格子是否全為指定圖塊。

        原理：
        - 檢查 (x, y) 上下左右四個格子是否全為指定圖塊。
        - 用於牆壁生成檢查。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            wall_type (list): 指定的圖塊類型列表。

        Returns:
            bool: 是否全為指定圖塊。
        """
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if self.xy_valid(nx, ny) and self.get_tile(nx, ny) not in wall_type:
                return False
        return True
    
    def _check_if_four_corners_all_wall_type(self, x, y, wall_type):
        """
        檢查 (x, y) 四個對角格子是否全為指定圖塊。

        原理：
        - 檢查 (x, y) 四個對角格子（左上、右上、左下、右下）是否全為指定圖塊。
        - 用於牆壁生成檢查。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            wall_type (list): 指定的圖塊類型列表。

        Returns:
            bool: 是否全為指定圖塊。
        """
        corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in corners:
            nx, ny = x + dx, y + dy
            if self.xy_valid(nx, ny) and self.get_tile(nx, ny) not in wall_type:
                return False
        return True
                
    def _check_if_surrounding_has_wall_type(self, x, y, wall_type):
        """
        檢查 (x, y) 周圍九宮格是否至少有一個指定圖塊。

        原理：
        - 檢查 (x, y) 格子周圍的八個格子（3x3 區域，排除中心點），是否至少有一個為指定圖塊。
        - 用於牆壁擴展時檢查連續性。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            wall_type (list): 指定的圖塊類型列表。

        Returns:
            bool: 是否至少有一個指定圖塊。
        """
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) in wall_type:
                    return True
        return False

    def _check_if_four_edges_has_wall_type(self, x, y, wall_type):
        """
        檢查 (x, y) 四個邊緣格子是否全無指定圖塊。

        原理：
        - 檢查 (x, y) 上下左右四個格子是否全無指定圖塊。
        - 用於牆壁生成檢查。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            wall_type (list): 指定的圖塊類型列表。

        Returns:
            bool: 是否全無指定圖塊。
        """
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if self.xy_valid(nx, ny) and self.get_tile(nx, ny) in wall_type:
                return False
        return True
    
    def _check_if_four_corners_has_wall_type(self, x, y, wall_type):
        """
        檢查 (x, y) 四個對角格子是否全無指定圖塊。

        原理：
        - 檢查 (x, y) 四個對角格子是否全無指定圖塊。
        - 用於牆壁生成檢查。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            wall_type (list): 指定的圖塊類型列表。

        Returns:
            bool: 是否全無指定圖塊。
        """
        corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in corners:
            nx, ny = x + dx, y + dy
            if self.xy_valid(nx, ny) and self.get_tile(nx, ny) in wall_type:
                return False
        return True
    
    def _check_if_five_square_has_wall_type(self, x, y, wall_type):
        """
        檢查 (x, y) 周圍 5x5 區域是否至少有一個指定圖塊。

        原理：
        - 檢查 (x, y) 格子周圍的 5x5 區域（排除中心點），是否至少有一個為指定圖塊。
        - 用於牆壁生成時確保牆壁分佈均勻。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。
            wall_type (list): 指定的圖塊類型列表。

        Returns:
            bool: 是否至少有一個指定圖塊。
        """
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.xy_valid(nx, ny) and self.get_tile(nx, ny) in wall_type:
                    return True
        return False
        
    def _get_connected_wall_size(self, x, y):
        """
        計算 (x, y) 所在連通牆壁組的大小。

        原理：
        - 使用洪水填充演算法，計算以 (x, y) 為起點的連通臨時牆壁（TILE_TEMP_WALL）的大小。
        - 用於限制牆壁擴展的連續性，防止過長的牆壁影響迷宮可玩性。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。

        Returns:
            int: 連通牆壁組的大小。
        """
        size, _ = self._flood_fill(x, y, TILE_TEMP_WALL)
        return size

    def convert_all_T_and_A_to_wall(self):
        """
        將所有臨時圖塊（TILE_TEMP_WALL 和 TILE_TEMP_MARKER）轉換為牆壁。

        原理：
        - 在牆壁擴展和路徑縮窄後，將臨時標記的格子統一轉換為牆壁（TILE_WALL）。
        - 確保最終迷宮結構清晰，所有牆壁使用統一的圖塊類型。
        """
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) in [TILE_TEMP_WALL, TILE_TEMP_MARKER]:
                    self.set_tile(x, y, TILE_WALL)
    
    def if_dead_end(self, x, y):
        """
        檢查 (x, y) 是否為死路（三面或以上被牆壁包圍）。

        原理：
        - 檢查 (x, y) 格子是否為路徑，且其四個方向（上下左右）中至少有三個方向被非路徑格子（牆壁、邊界等）阻擋。
        - 死路影響遊戲體驗，因此在牆壁生成時避免創建。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。

        Returns:
            bool: 是否為死路。
        """
        if not self.xy_valid(x, y) or self.get_tile(x, y) != TILE_PATH:
            return False
        return sum(1 for dx, dy in self.directions if self.get_tile(x + dx, y + dy) in [TILE_BOUNDARY, TILE_WALL]) >= 3

    def check_if_surrounding_no_dead_end(self, x, y):
        """
        檢查 (x, y) 周圍九宮格是否無死路。

        原理：
        - 檢查 (x, y) 周圍 3x3 區域內的每個格子，確保沒有死路。
        - 用於牆壁生成時避免創建死路。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。

        Returns:
            bool: 是否無死路。
        """
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if self.if_dead_end(x + dx, y + dy):
                    return False
        return True
    
    def extend_walls(self, extend_prob=0.99):
        """
        以指定概率在現有牆壁的上下左右生成新牆壁，確保不產生死路。

        原理：
        - 在迷宮左半部分隨機選擇路徑格子，設置為牆壁（TILE_WALL）。
        - 以概率 extend_prob 向隨機方向擴展牆壁，使用臨時牆壁（TILE_TEMP_WALL）標記。
        - 檢查擴展是否形成死路，若安全則繼續擴展（最多 3 格），否則回退。
        - 限制擴展嘗試次數（max_attempts=1000），防止無限循環。
        - 僅在迷宮左半部分（x <= width // 2）生成牆壁，後續鏡像到右半部分。

        Args:
            extend_prob (float): 擴展牆壁的概率（預設 0.99）。
        """
        half_width = self.width // 2
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            attempts += 1
            # 收集適合生成牆壁的格子（周圍九宮格全為路徑且 5x5 內有牆壁或邊界）
            wall_positions = [(x, y) for y in range(1, self.height - 1) 
                             for x in range(1, half_width + 1) if self._check_if_surrounding_all_wall_type(x, y, [TILE_PATH]) and self._check_if_five_square_has_wall_type(x, y, [TILE_BOUNDARY, TILE_WALL])]
            if not wall_positions:
                break
            
            # 隨機選擇一個格子
            x, y = random.choice(wall_positions)
            self.set_tile(x, y, TILE_TEMP_WALL)
            connected_size = 1
            # 以概率 extend_prob 決定是否擴展
            if random.random() > extend_prob:
                continue
            # 隨機選擇一個擴展方向
            direction = random.choice(self.directions)
            new_x, new_y = x + direction[0], y + direction[1]
            tries = 1
            while True:
                # 設置新格子為臨時牆壁
                self.set_tile(new_x, new_y, TILE_TEMP_WALL)
                # 檢查新牆壁是否安全（無死路且周圍無其他牆壁）
                if not self._check_if_surrounding_has_wall_type(new_x, new_y, [TILE_WALL, TILE_BOUNDARY]) and self.check_if_surrounding_no_dead_end(new_x, new_y):
                    connected_size += 1
                    if connected_size > 3:  # 限制連續牆壁長度
                        break
                    # 根據嘗試次數調整擴展概率
                    if random.random() < (extend_prob / max(1, 1 + tries / 10.0)):
                        direction = random.choice(self.directions)
                        x, y = new_x, new_y
                        new_x, new_y = x + direction[0], y + direction[1]
                        tries += 1
                    else:
                        break
                elif tries <= 20:
                    # 回退為路徑並嘗試新方向
                    self.set_tile(new_x, new_y, TILE_PATH)
                    random.seed(tries + self.seed if self.seed else 0)
                    direction = random.choice(self.directions)
                    random.seed(self.seed if self.seed else 0)
                    new_x, new_y = x + direction[0], y + direction[1]
                    tries += 1
                else:
                    # 嘗試次數過多，回退並結束
                    self.set_tile(new_x, new_y, TILE_PATH)
                    break
            # 將臨時牆壁轉換為最終牆壁
            self.convert_all_T_and_A_to_wall()
    
    def _check_connectivity(self, area, blocked_cell=None):
        """
        檢查區域內格子是否連通，允許臨時阻擋一個格子。

        原理：
        - 使用洪水填充演算法，從區域內任意格子開始，檢查是否能訪問所有指定格子。
        - 若提供 blocked_cell，臨時將其設置為牆壁，模擬阻擋後的連通性。
        - 用於確保牆壁生成不會斷開迷宮的可達性。

        Args:
            area (set): 要檢查連通性的格子集合。
            blocked_cell (tuple, optional): 臨時阻擋的格子坐標 (x, y)。

        Returns:
            bool: 區域是否連通。
        """
        if not area:
            return True
        start = list(area)[0]
        if blocked_cell:
            original_tile = self.get_tile(*blocked_cell)
            self.set_tile(*blocked_cell, TILE_WALL)
        _, reachable = self._flood_fill(*start, TILE_PATH)
        if blocked_cell:
            self.set_tile(*blocked_cell, original_tile)
        return all(pos in reachable for pos in area)

    def _check_if_suitable_narrow(self, x, y):
        """
        檢查 (x, y) 是否適合縮窄路徑。

        原理：
        - 確保 (x, y) 是牆壁，且其對角格子的連通牆壁組與當前格子相同。
        - 防止縮窄時破壞迷宮結構。

        Args:
            x (int): x 坐標。
            y (int): y 坐標。

        Returns:
            bool: 是否適合縮窄。
        """
        # 確認 (x, y) 為牆壁
        if not self.xy_valid(x, y) or self.get_tile(x, y) != TILE_WALL:
            return False

        # 獲取 (x, y) 所在的連通牆壁組
        _, current_wall_group = self._flood_fill(x, y, TILE_WALL)

        # 定義四個對角方向
        corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # 檢查每個對角格子
        for dx, dy in corners:
            nx, ny = x + dx, y + dy
            if self.xy_valid(nx, ny) and self.get_tile(nx, ny) == TILE_WALL:
                # 獲取對角格子的連通牆壁組
                _, neighbor_wall_group = self._flood_fill(nx, ny, TILE_WALL)
                # 若對角格子不屬於同一組，則不適合縮窄
                if (nx, ny) not in current_wall_group:
                    return False

        return True
    
    def narrow_paths(self):
        """
        縮窄 2x2 的路徑區域，隨機選擇一個 2x2 路徑塊進行處理。

        原理：
        - 遍歷迷宮，找出所有 2x2 的路徑區域（四個格子均為 TILE_PATH）。
        - 將所有符合條件的 2x2 塊收集到列表，隨機選擇一個進行處理。
        - 對選定的 2x2 塊，隨機嘗試在四個點放置牆壁，檢查是否在 3x3 九宮格內形成死路。
        - 若無死路且 connectednum <= 5，則保留牆壁，否則回退。
        - 重複執行直到無法縮窄更多路徑。
        - 增加迷宮複雜度，減少寬敞的路徑區域。

        Returns:
            int: 成功放置的牆壁數量。
        """
        count = 0
        max_attempts = 1000  # 防止無限循環
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            # 收集所有 2x2 路徑區域
            blocks = []
            for y in range(1, self.height - 2):
                for x in range(1, self.width - 2):
                    if (self.get_tile(x, y) == TILE_PATH and
                        self.get_tile(x + 1, y) == TILE_PATH and
                        self.get_tile(x, y + 1) == TILE_PATH and
                        self.get_tile(x + 1, y + 1) == TILE_PATH):
                        blocks.append((x, y))

            if not blocks:
                break

            # 隨機選擇一個 2x2 塊
            start_x, start_y = random.choice(blocks)
            block = [(start_x, start_y), (start_x + 1, start_y), (start_x, start_y + 1), (start_x + 1, start_y + 1)]
            random.shuffle(block)

            placed = False
            for bx, by in block:
                self.set_tile(bx, by, TILE_WALL)
                connected_num, _ = self._flood_fill(bx, by, TILE_WALL)
                
                # 檢查是否無死路、連通牆壁數量適中且適合縮窄
                if self.check_if_surrounding_no_dead_end(bx, by) and connected_num <= 5 and self._check_if_suitable_narrow(bx, by):
                    placed = True
                    count += 1
                    break
                else:
                    self.set_tile(bx, by, TILE_PATH)

            if not placed:
                continue

        return count

    def place_power_pellets(self):
        """
        在迷宮中均勻放置能量球，數量根據空地數量動態調整，確保分佈均勻。

        原理：
        - 收集所有路徑格子，排除鬼魂重生點及其周圍格子。
        - 根據迷宮大小動態計算能量球數量：num_pellets = max(4, min(0.1 * empty_count, 16))。
        - 將迷宮分成 5x5 網格，計算每個網格的中心點。
        - 選擇分佈均勻的網格（最大化網格間距離），在每個網格中隨機放置一個能量球。
        - 若能量球數量不足，隨機補充剩餘數量。
        - 使用獨立種子（seed + 1000）確保能量球位置可重現但與牆壁生成獨立。

        Returns:
            int: 放置的能量球數量。
        """
        # 使用獨立種子，確保能量球放置與牆壁生成分離
        if self.seed is not None:
            random.seed(self.seed + 1000)

        # 收集有效候選格子（排除鬼魂重生點及其周圍）
        empty_cells = []
        exclude_cells = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == TILE_GHOST_SPAWN:
                    exclude_cells.add((x, y))
                    for dx, dy in self.directions:
                        nx, ny = x + dx, y + dy
                        if self.xy_valid(nx, ny):
                            exclude_cells.add((nx, ny))

        all_candidate_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.get_tile(x, y) == TILE_PATH and (x, y) not in exclude_cells:
                    all_candidate_cells.append((x, y))

        empty_count = len(all_candidate_cells)

        # 動態計算能量球數量（4到16個）
        num_pellets = max(4, min(int(empty_count * 0.1), 16))

        pellet_positions = set()

        if empty_count >= num_pellets:
            # 將迷宮分成 5x5 網格
            grid_size_x = max(1, self.width // 5)
            grid_size_y = max(1, self.height // 5)
            grid_cells_map = {}

            # 分配候選格子到網格
            for cell in all_candidate_cells:
                x, y = cell
                gx = x // grid_size_x
                gy = y // grid_size_y
                grid_key = (gx, gy)
                if grid_key not in grid_cells_map:
                    grid_cells_map[grid_key] = []
                grid_cells_map[grid_key].append(cell)

            # 計算每個網格的中心點
            grid_centers = []
            for grid_key in grid_cells_map:
                gx, gy = grid_key
                cells = grid_cells_map[grid_key]
                if cells:
                    avg_x = sum(x for x, _ in cells) / len(cells)  # 網格內格子的平均 x 坐標
                    avg_y = sum(y for _, y in cells) / len(cells)  # 網格內格子的平均 y 坐標
                    grid_centers.append((grid_key, (avg_x, avg_y)))

            # 選擇分佈均勻的網格
            selected_grids = []
            selected_centers = []
            for _ in range(min(len(grid_centers), num_pellets)):
                if not grid_centers:
                    break
                if not selected_grids:
                    grid_key, center = random.choice(grid_centers)
                    selected_grids.append(grid_key)
                    selected_centers.append(center)
                    grid_centers = [(gk, c) for gk, c in grid_centers if gk != grid_key]
                else:
                    # 選擇距離現有網格中心最遠的網格
                    max_dist = -1
                    best_grid = None
                    best_center = None
                    for grid_key, center in grid_centers:
                        min_dist = float('inf')
                        for selected_center in selected_centers:
                            dist = ((center[0] - selected_center[0]) ** 2 + (center[1] - selected_center[1]) ** 2) ** 0.5
                            min_dist = min(min_dist, dist)
                        if min_dist > max_dist:
                            max_dist = min_dist
                            best_grid = grid_key
                            best_center = center
                    if best_grid:
                        selected_grids.append(best_grid)
                        selected_centers.append(best_center)
                        grid_centers = [(gk, c) for gk, c in grid_centers if gk != best_grid]

            # 在選定網格中隨機放置能量球
            for grid_key in selected_grids:
                if grid_cells_map[grid_key]:
                    pellet_positions.add(random.choice(grid_cells_map[grid_key]))

            # 補充剩餘能量球
            remaining_needed = num_pellets - len(pellet_positions)
            if remaining_needed > 0:
                available_for_random = [cell for cell in all_candidate_cells if cell not in pellet_positions]
                if available_for_random:
                    pellet_positions.update(random.sample(available_for_random, min(remaining_needed, len(available_for_random))))
        elif empty_count > 0:
            pellet_positions.update(random.sample(all_candidate_cells, empty_count))

        # 放置能量球
        for x, y in pellet_positions:
            self.set_tile(x, y, TILE_POWER_PELLET)

        # 恢復原始種子
        if self.seed is not None:
            random.seed(self.seed)

        return len(pellet_positions)

    def generate_maze(self):
        """
        生成完整的迷宮，包含牆壁擴展、路徑縮窄和能量球放置。
        左半部分生成後鏡像到右半部分，確保對稱性。

        原理：
        - 按以下步驟生成迷宮：
          1. add_central_room()：添加中央 5x5 房間。
          2. extend_walls()：在左半部分隨機生成牆壁。
          3. narrow_paths()：縮窄 2x2 路徑區域，增加複雜度。
          4. convert_all_T_and_A_to_wall()：將臨時牆壁轉換為最終牆壁。
          5. place_power_pellets()：均勻放置能量球。
        - 最後將左半部分（x < width // 2）的格子鏡像到右半部分，實現左右對稱。
        - 對稱性公式：右半部分格子 (width - 1 - x, y) = 左半部分格子 (x, y)。
        """
        self.add_central_room()  # 添加中央房間
        self.extend_walls()  # 生成左半部分牆壁
        
        while self.narrow_paths():  # 縮窄路徑直到無法繼續
            pass
        self.convert_all_T_and_A_to_wall()  # 轉換臨時牆壁
        self.place_power_pellets()  # 放置能量球
        half_width = self.width // 2
        # 將左半部分鏡像到右半部分
        for y in range(self.height):
            for x in range(0, half_width):
                self.set_tile(self.width - 1 - x, y, self.get_tile(x, y))

if __name__ == "__main__":
    width, height, seed = MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

    # 檢查迷宮尺寸是否足夠容納中央房間
    if width < 7 or height < 7:
        print("錯誤：迷宮最小尺寸為 7x7 以容納中央房間")
        sys.exit(1)
    
    maze = Map(width, height, seed=seed)
    maze.generate_maze()
    print(f"生成的 Pac-Man 迷宮（種子大小：{seed if seed is not None else '無'}）")
    print(f"尺寸：{width}x{height}")
    print(maze)