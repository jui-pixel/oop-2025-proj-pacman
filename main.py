# main.py
"""
Pac-Man 遊戲的主程式，負責初始化遊戲、處理事件、更新狀態和渲染畫面。
使用 Pygame 作為遊戲引擎，整合遊戲邏輯與 AI 控制策略。
"""
import sys
import pygame
import math
from game.game import Game
from ai.strategies import ControlManager
from config import *

# 初始化 Pygame
pygame.init()

def main():
    """
    主遊戲入口，負責設置遊戲環境、運行主迴圈和處理遊戲結束。
    """
    # 初始化遊戲實例
    game = Game()
    
    # 設置螢幕尺寸
    screen_width = MAZE_WIDTH * CELL_SIZE
    screen_height = MAZE_HEIGHT * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Game")
    
    # 設置遊戲時鐘和字體
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    
    # 初始化操控管理器，負責玩家和 AI 控制
    control_manager = ControlManager(MAZE_WIDTH, MAZE_HEIGHT)
    frame_count = 0  # 用於動畫效果的計數器
    
    # 主遊戲迴圈
    while game.running:
        frame_count += 1
        
        # 處理 Pygame 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False  # 關閉遊戲
            control_manager.handle_event(event)  # 處理鍵盤輸入或切換模式
        
        # 更新遊戲狀態
        game.update(FPS, lambda: control_manager.move(
            game.pacman, game.maze, game.power_pellets, game.score_pellets, game.ghosts))
        
        # 渲染遊戲畫面
        screen.fill(BLACK)  # 清空畫面
        
        # 渲染迷宮
        for y in range(game.maze.h):
            for x in range(game.maze.w):
                tile = game.maze.get_tile(x, y)
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if tile == '#':
                    pygame.draw.rect(screen, DARK_GRAY, rect)  # 繪製邊界
                elif tile == 'X':
                    pygame.draw.rect(screen, BLACK, rect)  # 繪製牆壁
                elif tile == '.':
                    pygame.draw.rect(screen, GRAY, rect)  # 繪製路徑
                elif tile == 'E':
                    pygame.draw.rect(screen, GREEN, rect)  # 繪製能量球位置
                elif tile == 'S':
                    pygame.draw.rect(screen, PINK, rect)  # 繪製鬼魂重生點
                elif tile == 'D':
                    pygame.draw.rect(screen, RED, rect)  # 繪製門
       
        # 渲染能量球
        for pellet in game.power_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * CELL_SIZE + CELL_SIZE // 4,
                pellet.y * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.ellipse(screen, BLUE, pellet_rect)
        
        # 渲染分數球
        for score_pellet in game.score_pellets:
            score_pellet_rect = pygame.Rect(
                score_pellet.x * CELL_SIZE + CELL_SIZE // 4,
                score_pellet.y * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.ellipse(screen, ORANGE, score_pellet_rect)
        
        # 渲染 Pac-Man
        pacman_rect = pygame.Rect(
            game.pacman.current_x - CELL_SIZE // 4,
            game.pacman.current_y - CELL_SIZE // 4,
            CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.ellipse(screen, YELLOW, pacman_rect)
        
        # 渲染鬼魂
        for ghost in game.ghosts:
            if ghost.returning_to_spawn:
                base_color = DARK_GRAY
                ghost.alpha = int(128 + 127 * math.sin(frame_count * 0.2))  # 閃爍效果
            elif ghost.edible and ghost.edible_timer > 0:
                base_color = LIGHT_BLUE
                ghost.alpha = 255
            else:
                base_color = ghost.color
                ghost.alpha = 255
            
            ghost_surface = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2), pygame.SRCALPHA)
            ghost_surface.fill((0, 0, 0, 0))  # 透明背景
            pygame.draw.ellipse(ghost_surface, (*base_color, ghost.alpha),
                               (0, 0, CELL_SIZE // 2, CELL_SIZE // 2))
            screen.blit(ghost_surface, (ghost.current_x - CELL_SIZE // 4, ghost.current_y - CELL_SIZE // 4))
        
        # 渲染分數和控制模式
        score_text = font.render(f"Score: {game.pacman.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        mode_text = font.render(control_manager.get_mode_name(), True, WHITE)
        screen.blit(mode_text, (screen_width - 150, 10))
        
        pygame.display.flip()  # 更新畫面
        clock.tick(FPS)  # 控制幀率
    
    # 遊戲結束，清理並退出
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()