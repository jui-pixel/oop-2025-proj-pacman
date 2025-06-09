# game/renderer.py
"""
負責渲染遊戲畫面，包括迷宮、Pac-Man、鬼魂和分數顯示。
"""

import pygame
import math
from .entities.pellets import PowerPellet, ScorePellet
from .entities.pacman import PacMan
from .entities.ghost import Ghost
from typing import List, Tuple

from .maze_generator import Map
from config import BLACK, DARK_GRAY, GRAY, GREEN, PINK, RED, BLUE, ORANGE, YELLOW, WHITE, LIGHT_BLUE, CELL_SIZE, TILE_BOUNDARY, TILE_WALL, TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN, TILE_DOOR
from .game import Game

class Renderer:
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font, screen_width: int, screen_height: int):
        """
        初始化渲染器。

        原理：
        - 設置 Pygame 畫面物件、字體和螢幕尺寸，用於後續的圖形和文字渲染。
        - 儲存這些參數以便在 render 方法中重複使用。

        Args:
            screen (pygame.Surface): Pygame 畫面物件，用於繪製遊戲畫面。
            font (pygame.font.Font): 用於渲染文字的字體（分數、生命值、控制模式等）。
            screen_width (int): 螢幕寬度（像素）。
            screen_height (int): 螢幕高度（像素）。
        """
        self.screen = screen
        self.font = font
        self.screen_width = screen_width
        self.screen_height = screen_height

    def render(self, game: 'Game', control_mode: str, frame_count: int) -> None:
        """
        渲染遊戲畫面。

        原理：
        - 每幀調用一次，負責繪製迷宮、能量球、分數球、Pac-Man、鬼魂以及分數、生命值和控制模式文字。
        - 使用 Pygame 的繪圖函數（如 draw.rect、draw.ellipse、blit）根據遊戲狀態渲染對應圖形。
        - 支援動畫效果，例如 Pac-Man 的死亡縮小動畫和鬼魂的閃爍效果。
        - 坐標計算公式：像素坐標 = 格子坐標 * CELL_SIZE + 偏移量（如 CELL_SIZE // 4）。

        Args:
            game (Game): 遊戲實例，提供迷宮、Pac-Man、鬼魂等數據。
            control_mode (str): 當前控制模式名稱（例如 "Player Mode"）。
            frame_count (int): 動畫幀計數器，用於控制動畫效果（如鬼魂閃爍）。
        """
        self.screen.fill(BLACK)  # 清空畫面，設置背景為黑色

        # 渲染迷宮
        maze = game.get_maze()
        for y in range(maze.height):
            for x in range(maze.width):
                tile = maze.get_tile(x, y)
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)  # 計算格子矩形
                if tile == TILE_BOUNDARY:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)  # 繪製邊界（深灰色）
                elif tile == TILE_WALL:
                    pygame.draw.rect(self.screen, BLACK, rect)  # 繪製牆壁（黑色）
                elif tile == TILE_PATH:
                    pygame.draw.rect(self.screen, GRAY, rect)  # 繪製路徑（灰色）
                elif tile == TILE_POWER_PELLET:
                    pygame.draw.rect(self.screen, GREEN, rect)  # 繪製能量球位置（綠色）
                elif tile == TILE_GHOST_SPAWN:
                    pygame.draw.rect(self.screen, PINK, rect)  # 繪製鬼魂重生點（粉紅色）
                elif tile == TILE_DOOR:
                    pygame.draw.rect(self.screen, RED, rect)  # 繪製門（紅色）

        # 渲染能量球
        for pellet in game.get_power_pellets():
            pellet_rect = pygame.Rect(
                pellet.x * CELL_SIZE + CELL_SIZE // 4,
                pellet.y * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2, CELL_SIZE // 2)  # 計算能量球矩形（居中，半格大小）
            pygame.draw.ellipse(self.screen, BLUE, pellet_rect)  # 繪製藍色圓形能量球

        # 渲染分數球
        for score_pellet in game.get_score_pellets():
            score_pellet_rect = pygame.Rect(
                score_pellet.x * CELL_SIZE + CELL_SIZE // 4,
                score_pellet.y * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2, CELL_SIZE // 2)  # 計算分數球矩形（居中，半格大小）
            pygame.draw.ellipse(self.screen, ORANGE, score_pellet_rect)  # 繪製橙色圓形分數球

        # 渲染 Pac-Man
        pacman = game.get_pacman()
        
        if game.is_death_animation_playing():
            # 死亡動畫：縮小 Pac-Man
            progress = game.get_death_animation_progress()  # 動畫進度（0 到 1）
            scale = 1.0 - progress  # 縮放比例，從 1 減小到 0
            radius = int(CELL_SIZE // 2 * scale)  # 計算縮放後的圓半徑
            pacman_center = (
                pacman.current_x,
                pacman.current_y,
                )  # Pac-Man 當前中心坐標
            if radius > 0:
                pygame.draw.circle(self.screen, YELLOW, pacman_center, radius)  # 繪製縮小的黃色圓形
        else:
            # 正常繪製 Pac-Man
            pacman_rect = pygame.Rect(
                pacman.current_x - CELL_SIZE // 4,
                pacman.current_y - CELL_SIZE // 4,
                CELL_SIZE // 2, CELL_SIZE // 2)  # 計算 Pac-Man 矩形（居中，半格大小）
            pygame.draw.ellipse(self.screen, YELLOW, pacman_rect)  # 繪製黃色圓形 Pac-Man

        # 渲染鬼魂
        for ghost in game.get_ghosts():
            if ghost.returning_to_spawn:
                base_color = DARK_GRAY  # 返回重生點時為深灰色
                ghost.alpha = int(128 + 127 * math.sin(frame_count * 0.5))  # 閃爍效果，透明度 128~255
            elif ghost.edible and ghost.edible_timer > 0:
                base_color = LIGHT_BLUE  # 可食用狀態為淺藍色
                ghost.alpha = 255  # 完全不透明
            else:
                base_color = ghost.color  # 正常狀態使用鬼魂的原始顏色
                ghost.alpha = 255  # 完全不透明

            ghost_surface = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2), pygame.SRCALPHA)  # 創建透明表面
            ghost_surface.fill((0, 0, 0, 0))  # 設置透明背景
            pygame.draw.ellipse(ghost_surface, (*base_color, ghost.alpha),
                               (0, 0, CELL_SIZE // 2, CELL_SIZE // 2))  # 繪製鬼魂圓形
            self.screen.blit(ghost_surface, (ghost.current_x - CELL_SIZE // 4, ghost.current_y - CELL_SIZE // 4))  # 將鬼魂繪製到螢幕

        # 渲染分數、控制模式和生命值
        score_text = self.font.render(f"Score: {pacman.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))  # 顯示分數（左上角）
        lives_text = self.font.render(f"Lives: {game.get_lives()}", True, WHITE)
        self.screen.blit(lives_text, (10, 50))  # 顯示生命值（左上角下方）
        mode_text = self.font.render(control_mode, True, WHITE)
        self.screen.blit(mode_text, (self.screen_width - 150, 10))  # 顯示控制模式（右上角）