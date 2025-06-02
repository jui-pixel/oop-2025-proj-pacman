# game/entities/entity_base.py
"""
定義遊戲中實體的基類，提供基本移動和目標設置功能。
"""
from typing import Tuple
from config import CELL_SIZE, TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR, ENTITY_DEFAULT_SPEED

class Entity:
    def __init__(self, x: int, y: int, symbol: str):
        """
        初始化基本實體，設置位置和符號。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            symbol (str): 實體的符號表示。
        """
        self.x = x
        self.y = y
        self.symbol = symbol
        self.target_x = x
        self.target_y = y
        self.current_x = x * CELL_SIZE + CELL_SIZE // 2  # 像素座標
        self.current_y = y * CELL_SIZE + CELL_SIZE // 2
        self.speed = ENTITY_DEFAULT_SPEED  # 移動速度（像素/幀）

    def move_towards_target(self) -> bool:
        """
        逐像素移動到目標格子，防止速度溢出。
        
        Returns:
            bool: 是否到達目標格子。
        """
        target_pixel_x = self.target_x * CELL_SIZE + CELL_SIZE // 2
        target_pixel_y = self.target_y * CELL_SIZE + CELL_SIZE // 2
        dx = target_pixel_x - self.current_x
        dy = target_pixel_y - self.current_y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        
        if dist <= self.speed:
            self.current_x = target_pixel_x
            self.current_y = target_pixel_y
            self.x = self.target_x
            self.y = self.target_y
            return True
        else:
            if dist != 0:
                move_dist = min(self.speed, dist)
                self.current_x += (dx / dist) * move_dist
                self.current_y += (dy / dist) * move_dist
            return False

    def set_new_target(self, dx: int, dy: int, maze) -> bool:
        """
        設置新目標格子，檢查是否可通行。
        
        Args:
            dx (int): x 方向偏移。
            dy (int): y 方向偏移。
            maze: 迷宮物件。
        
        Returns:
            bool: 是否成功設置目標。
        """
        new_x, new_y = self.x + dx, self.y + dy
        if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in [TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR]:
            self.target_x, self.target_y = new_x, new_y
            return True
        return False