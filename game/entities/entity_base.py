# game/entities/entity_base.py
"""
定義遊戲中實體的基類，提供基本的移動和目標設置功能，適用於 Pac-Man、鬼魂和彈丸等實體。
"""

from typing import Tuple
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
        self.x = x  # 格子 x 坐標
        self.y = y  # 格子 y 坐標
        self.symbol = symbol  # 實體符號
        self.target_x = x  # 目標格子 x 坐標
        self.target_y = y  # 目標格子 y 坐標
        self.current_x = x * CELL_SIZE + CELL_SIZE // 2  # 像素 x 坐標（格子中心）
        self.current_y = y * CELL_SIZE + CELL_SIZE // 2  # 像素 y 坐標（格子中心）
        self.speed = ENTITY_DEFAULT_SPEED  # 移動速度（像素/幀）

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
        target_pixel_x = self.target_x * CELL_SIZE + CELL_SIZE // 2  # 目標像素 x 坐標
        target_pixel_y = self.target_y * CELL_SIZE + CELL_SIZE // 2  # 目標像素 y 坐標
        dx = target_pixel_x - self.current_x  # x 方向距離
        dy = target_pixel_y - self.current_y  # y 方向距離
        dist = (dx ** 2 + dy ** 2) ** 0.5  # 歐幾里得距離

        if dist <= self.speed / fps:  # 若距離小於每幀移動距離，則到達目標
            self.current_x = target_pixel_x
            self.current_y = target_pixel_y
            self.x = self.target_x
            self.y = self.target_y
            return True
        else:
            if dist != 0:  # 避免除以零
                move_dist = min(self.speed / fps, dist)  # 每幀移動距離
                self.current_x += (dx / dist) * move_dist  # 更新 x 坐標
                self.current_y += (dy / dist) * move_dist  # 更新 y 坐標
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
        new_x, new_y = self.x + dx, self.y + dy
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR]:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False