# main.py
import sys
import pygame
from game.maze_generator import Map
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
import random
import math

# Pygame 初始化
pygame.init()

# 遊戲參數
CELL_SIZE = 20
FPS = 30

# 顏色定義
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
LIGHT_BLUE = (173, 216, 230)
DARK_GRAY = (50, 50, 50)

def main():
    # 生成迷宮
    maze = Map(w=29, h=31, seed=1)
    maze.generate_connected_maze(path_density=0.7)

    # 初始化遊戲角色
    pacman, ghosts, power_pellets, score_pellets = initialize_entities(maze)
    respawn_points = [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']

    # 設置 Pygame 視窗
    screen_width = maze.w * CELL_SIZE
    screen_height = maze.h * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Game")
    clock = pygame.time.Clock()

    # 初始化字體
    font = pygame.font.SysFont(None, 36)

    # 遊戲參數
    edible_duration = 300
    ghost_scores = [50, 100, 150, 200]
    ghost_score_index = 0
    frame_count = 0

    # 遊戲主循環
    running = True
    moving = False
    while running:
        frame_count += 1
        # 處理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                dx, dy = 0, 0
                if event.key == pygame.K_UP:
                    dx, dy = 0, -1
                elif event.key == pygame.K_DOWN:
                    dx, dy = 0, 1
                elif event.key == pygame.K_LEFT:
                    dx, dy = -1, 0
                elif event.key == pygame.K_RIGHT:
                    dx, dy = 1, 0
                if not moving and (dx != 0 or dy != 0):
                    if pacman.set_new_target(dx, dy, maze):
                        moving = True

        # 移動 Pac-Man
        if moving:
            if pacman.move_towards_target(maze):
                moving = False

        # 檢查 Pac-Man 是否吃到能量球
        score_from_pellet = pacman.eat_pellet(power_pellets)
        if score_from_pellet > 0:
            for ghost in ghosts:
                ghost.set_edible(edible_duration)
            ghost_score_index = 0

        # 檢查 Pac-Man 是否吃到分數球
        pacman.eat_score_pellet(score_pellets)

        # 移動鬼魂
        for ghost in ghosts:
            if ghost.move_towards_target(maze):
                if ghost.returning_to_spawn and maze.get_tile(ghost.x, ghost.y) == 'S':
                    ghost.set_waiting(FPS)
                else:
                    ghost.move(pacman, maze, FPS)

        # 檢查 Pac-Man 是否與鬼魂碰撞
        collision_detected = False
        for ghost in ghosts:
            if pacman.x == ghost.x and pacman.y == ghost.y:
                collision_detected = True
                if ghost.edible and ghost.respawn_timer > 0:
                    pacman.score += ghost_scores[ghost_score_index]
                    ghost_score_index = min(ghost_score_index + 1, len(ghost_scores) - 1)
                    ghost.set_returning_to_spawn(FPS)
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    print(f"Game Over! Score: {pacman.score}")
                    running = False
                break
        if collision_detected and not running:
            break

        # 渲染迷宮
        screen.fill(BLACK)
        for y in range(maze.h):
            for x in range(maze.w):
                tile = maze.get_tile(x, y)
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if tile == '#':
                    pygame.draw.rect(screen, BLACK, rect)
                elif tile == 'X':
                    pygame.draw.rect(screen, BLACK, rect)
                elif tile == '.':
                    pygame.draw.rect(screen, GRAY, rect)
                elif tile == 'A':
                    pygame.draw.rect(screen, GREEN, rect)
                elif tile == 'S':
                    pygame.draw.rect(screen, GRAY, rect)

        # 渲染能量球
        for pellet in power_pellets:
            pellet_rect = pygame.Rect(pellet.x * CELL_SIZE + CELL_SIZE // 4,
                                     pellet.y * CELL_SIZE + CELL_SIZE // 4,
                                     CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.ellipse(screen, BLUE, pellet_rect)

        # 渲染分數球
        for score_pellet in score_pellets:
            score_pellet_rect = pygame.Rect(score_pellet.x * CELL_SIZE + CELL_SIZE // 4,
                                           score_pellet.y * CELL_SIZE + CELL_SIZE // 4,
                                           CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.ellipse(screen, ORANGE, score_pellet_rect)

        # 渲染 Pac-Man
        pacman_rect = pygame.Rect(pacman.current_x - CELL_SIZE // 4,
                                 pacman.current_y - CELL_SIZE // 4,
                                 CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.ellipse(screen, YELLOW, pacman_rect)

        # 渲染鬼魂
        for ghost in ghosts:
            if ghost.returning_to_spawn:
                base_color = DARK_GRAY
                ghost.alpha = int(128 + 127 * math.sin(frame_count * 0.2))
            else:
                base_color = LIGHT_BLUE if ghost.edible else RED
                ghost.alpha = 255

            ghost_surface = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2), pygame.SRCALPHA)
            ghost_surface.fill((0, 0, 0, 0))
            pygame.draw.ellipse(ghost_surface, (*base_color, ghost.alpha),
                               (0, 0, CELL_SIZE // 2, CELL_SIZE // 2))
            screen.blit(ghost_surface, (ghost.current_x - CELL_SIZE // 4, ghost.current_y - CELL_SIZE // 4))

        # 渲染分數
        score_text = font.render(f"Score: {pacman.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        # 更新顯示
        pygame.display.flip()
        clock.tick(FPS)

    # 結束遊戲
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()