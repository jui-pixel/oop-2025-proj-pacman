# main.py
import sys
import pygame
import math
from game.game import Game
from ai.strategies import ControlManager
from config import *

pygame.init()

def main():
    """主遊戲入口。"""
    # 初始化遊戲
    game = Game()
    screen_width = MAZE_WIDTH * CELL_SIZE
    screen_height = MAZE_HEIGHT * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    # 初始化操控管理器
    control_manager = ControlManager(MAZE_WIDTH, MAZE_HEIGHT)
    frame_count = 0

    while game.running:
        frame_count += 1

        # 處理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
            control_manager.handle_event(event)

        # 更新遊戲狀態
        game.update(FPS, lambda: control_manager.move(game.pacman, game.maze, game.power_pellets, game.score_pellets, game.ghosts))

        # 渲染
        screen.fill(BLACK)
        # 渲染迷宮
        for y in range(game.maze.h):
            for x in range(game.maze.w):
                tile = game.maze.get_tile(x, y)
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if tile == '#':
                    pygame.draw.rect(screen, DARK_GRAY, rect)
                elif tile == 'X':
                    pygame.draw.rect(screen, BLACK, rect)
                elif tile == '.':
                    pygame.draw.rect(screen, GRAY, rect)
                elif tile == 'E':
                    pygame.draw.rect(screen, GREEN, rect)
                elif tile == 'S':
                    pygame.draw.rect(screen, PINK, rect)
                elif tile == 'D':
                    pygame.draw.rect(screen, RED, rect)

        # 渲染能量球
        for pellet in game.power_pellets:
            pellet_rect = pygame.Rect(pellet.x * CELL_SIZE + CELL_SIZE // 4,
                                     pellet.y * CELL_SIZE + CELL_SIZE // 4,
                                     CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.ellipse(screen, BLUE, pellet_rect)

        # 渲染分數球
        for score_pellet in game.score_pellets:
            score_pellet_rect = pygame.Rect(score_pellet.x * CELL_SIZE + CELL_SIZE // 4,
                                           score_pellet.y * CELL_SIZE + CELL_SIZE // 4,
                                           CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.ellipse(screen, ORANGE, score_pellet_rect)

        # 渲染 Pac-Man
        pacman_rect = pygame.Rect(game.pacman.current_x - CELL_SIZE // 4,
                                 game.pacman.current_y - CELL_SIZE // 4,
                                 CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.ellipse(screen, YELLOW, pacman_rect)

        # 渲染鬼魂
        for ghost in game.ghosts:
            if ghost.returning_to_spawn:
                base_color = DARK_GRAY
                ghost.alpha = int(128 + 127 * math.sin(frame_count * 0.2))
            elif ghost.edible and ghost.edible_timer > 0:
                base_color = LIGHT_BLUE
                ghost.alpha = 255
            else:
                base_color = ghost.color  # 使用鬼魂的獨特顏色
                ghost.alpha = 255

            ghost_surface = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2), pygame.SRCALPHA)
            ghost_surface.fill((0, 0, 0, 0))
            pygame.draw.ellipse(ghost_surface, (*base_color, ghost.alpha),
                               (0, 0, CELL_SIZE // 2, CELL_SIZE // 2))
            screen.blit(ghost_surface, (ghost.current_x - CELL_SIZE // 4, ghost.current_y - CELL_SIZE // 4))

        # 渲染分數和模式
        score_text = font.render(f"Score: {game.pacman.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        mode_text = font.render(control_manager.get_mode_name(), True, WHITE)
        screen.blit(mode_text, (screen_width - 150, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()