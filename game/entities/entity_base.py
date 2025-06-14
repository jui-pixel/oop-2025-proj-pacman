# game/entities/entity_base.py
"""
定義遊戲中實體的基類，提供基本的移動和目標設置功能，適用於 Pac-Man、鬼魂和彈丸等實體。

這個基類為所有遊戲實體提供通用的屬性和方法，確保它們在迷宮中能夠正確移動和定位。
"""

# 匯入型別提示模組
from typing import Tuple
# 從 config 檔案匯入常數
from config import CELL_SIZE, TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR, ENTITY_DEFAULT_SPEED

class Entity:
    def __init__(self, x: int, y: int, symbol: str):
        """
        初始化基本實體，設置位置、符號和像素坐標。

        原理：
        - Entity 基類為所有遊戲實體（Pac-Man、鬼魂、彈丸）提供通用的屬性和方法。
        - 使用格子坐標 (x, y) 和像素坐標 (current_x, current_y) 分別表示邏輯位置和渲染位置。
        - 像素坐標計算公式：current_x = x * CELL_SIZE + CELL_SIZE // 2
        - 實體移動採用平滑的逐像素移動，確保視覺效果連貫。

        Args:
            x (int): 迷宮中的 x 坐標（格子坐標）。
            y (int): 迷宮中的 y 坐標（格子坐標）。
            symbol (str): 實體的符號表示（例如 'P' 表示 Pac-Man，'G' 表示鬼魂）。
        """
        # 設置格子坐標
        self.x = x
        self.y = y
        # 設置實體符號
        self.symbol = symbol
        # 設置初始目標格子坐標（與當前坐標相同）
        self.target_x = x
        self.target_y = y
        # 計算像素坐標（格子中心）
        self.current_x = x * CELL_SIZE + CELL_SIZE // 2
        self.current_y = y * CELL_SIZE + CELL_SIZE // 2
        # 設置移動速度（像素/幀）
        self.speed = ENTITY_DEFAULT_SPEED

    def move_towards_target(self, fps: int) -> bool:
        """
        逐像素移動到目標格子，防止速度溢出，實現平滑移動。

        原理：
        - 實體從當前像素坐標 (current_x, current_y) 向目標像素坐標 (target_pixel_x, target_pixel_y) 移動。
        - 距離計算公式：dist = √((target_pixel_x - current_x)^2 + (target_pixel_y - current_y)^2)
        - 每幀移動距離：move_dist = min(speed / fps, dist)，確保不超過目標點。
        - 若到達目標格子，更新格子坐標 (x, y) 並返回 True。

        Args:
            fps (int): 每秒幀數，用於計算每幀移動距離。

        Returns:
            bool: 是否到達目標格子。
        """
        # 計算目標像素坐標（目標格子中心）
        target_pixel_x = self.target_x * CELL_SIZE + CELL_SIZE // 2
        target_pixel_y = self.target_y * CELL_SIZE + CELL_SIZE // 2
        # 計算當前位置到目標的距離（x 和 y 方向）
        dx = target_pixel_x - self.current_x
        dy = target_pixel_y - self.current_y
        # 計算歐幾里得距離
        dist = (dx ** 2 + dy ** 2) ** 0.5

        # 如果距離小於每幀移動距離，表示已到達目標
        if dist <= self.speed / fps:
            # 更新像素坐標到目標位置
            self.current_x = target_pixel_x
            self.current_y = target_pixel_y
            # 更新格子坐標
            self.x = self.target_x
            self.y = self.target_y
            return True
        else:
            # 如果尚未到達，計算每幀應移動的距離
            if dist != 0:  # 避免除以零
                move_dist = min(self.speed / fps, dist)  # 限制移動距離不超過目標
                # 按比例更新像素坐標
                self.current_x += (dx / dist) * move_dist
                self.current_y += (dy / dist) * move_dist
            return False

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """
        設置新目標格子，檢查是否可通行。

        原理：
        - 檢查目標位置 (new_x, new_y) 是否在迷宮內有效且為可通行格子（路徑、能量球、鬼魂重生點或門）。
        - 可通行格子由迷宮的瓦片類型決定：TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR。
        - 若有效，更新目標坐標 (target_x, target_y) 並返回 True。

        Args:
            dx (int): x 方向偏移（例如 -1 表示左，1 表示右）。
            dy (int): y 方向偏移（例如 -1 表示上，1 表示下）。
            maze: 迷宮物件，提供瓦片信息和有效性檢查。

        Returns:
            bool: 是否成功設置目標。
        """
        # 計算新目標格子坐標
        new_x, new_y = self.x + dx, self.y + dy
        # 檢查目標是否有效且為可通行格子
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR]:
            # 更新目標坐標
            self.target_x, self.target_y = new_x, new_y
            return True
        return False