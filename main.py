# main.py
import sys
import pygame
from game.maze_generator import Map
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
# from ai.agent import DQNAgent
# import torch
import numpy as np
import random

pygame.init()

CELL_SIZE = 20
FPS = 30

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
    maze = Map(w=29, h=31, seed=1)
    maze.generate_connected_maze(path_density=0.7)

    pacman, ghosts, power_pellets, score_pellets = initialize_entities(maze)
    respawn_points = [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']

    screen_width = maze.w * CELL_SIZE
    screen_height = maze.h * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Game")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 36)

    edible_duration = 300
    ghost_scores = [50, 100, 150, 200]
    ghost_score_index = 0
    frame_count = 0

    # # 載入訓練好的 DQN 模型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # agent = DQNAgent((maze.w, maze.h, 5), 4, device)
    # agent.load("pacman_dqn_final.pth")  # 假設模型已訓練好並保存

    # 操控模式：False 表示玩家操控，True 表示 AI 操控
    use_ai = False
    moving = False

    running = True
    while running:
        frame_count += 1

        # 處理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # 按下 'A' 鍵切換操控模式
                if event.key == pygame.K_a:
                    use_ai = not use_ai
                    mode = "AI Mode" if use_ai else "Player Mode"
                    print(f"Switched to {mode}")

                # 玩家操控模式下，處理鍵盤輸入
                if not use_ai and not moving:
                    dx, dy = 0, 0
                    if event.key == pygame.K_UP:
                        dx, dy = 0, -1
                    elif event.key == pygame.K_DOWN:
                        dx, dy = 0, 1
                    elif event.key == pygame.K_LEFT:
                        dx, dy = -1, 0
                    elif event.key == pygame.K_RIGHT:
                        dx, dy = 1, 0
                    if dx != 0 or dy != 0:
                        if pacman.set_new_target(dx, dy, maze):
                            moving = True

        # AI 操控模式下，自動預測行動
        if use_ai and not moving:
            state = np.zeros((maze.w, maze.h, 5))
            state[pacman.x, pacman.y, 0] = 1
            for pellet in power_pellets:
                state[pellet.x, pellet.y, 1] = 1
            for pellet in score_pellets:
                state[pellet.x, pellet.y, 2] = 1
            for ghost in ghosts:
                if ghost.edible and ghost.respawn_timer > 0:
                    state[ghost.x, ghost.y, 3] = 1
                else:
                    state[ghost.x, ghost.y, 4] = 1
            action = agent.get_action(state)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
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

        # 渲染操控模式
        mode_text = font.render("AI Mode" if use_ai else "Player Mode", True, WHITE)
        screen.blit(mode_text, (screen_width - 150, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()