# main.py
import sys
import pygame
from game.maze_generator import Map
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities

# Pygame 初始化
pygame.init()

# 遊戲參數
CELL_SIZE = 20  # 每個迷宮格子的大小（像素）
FPS = 10  # 每秒幀數（遊戲速度）
GHOST_MOVE_INTERVAL = 2  # 鬼魂每隔多少幀移動一次（設為 2 讓鬼魂速度減半）

# 顏色定義
BLACK = (0, 0, 0)        # 背景色和牆壁
WHITE = (255, 255, 255)  # 分數文字
YELLOW = (255, 255, 0)   # Pac-Man
RED = (255, 0, 0)        # 鬼魂
BLUE = (0, 0, 255)       # 能量球
ORANGE = (255, 165, 0)   # 分數球
GRAY = (128, 128, 128)   # 路徑
GREEN = (0, 255, 0)      # 死路/偏僻區域 ('A')

def main():
    # 生成迷宮
    maze = Map(w=29, h=31, seed=1)
    maze.generate_connected_maze(path_density=0.7)

    # 初始化遊戲角色
    pacman, ghosts, power_pellets, score_pellets = initialize_entities(maze)

    # 設置 Pygame 視窗
    screen_width = maze.w * CELL_SIZE
    screen_height = maze.h * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Game")
    clock = pygame.time.Clock()

    # 初始化字體
    font = pygame.font.SysFont(None, 36)  # 使用系統字體，大小 36

    # 移動計數器
    frame_count = 0

    # 遊戲主循環
    running = True
    while running:
        # 處理事件（例如關閉視窗或鍵盤輸入）
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
                # 移動 Pac-Man
                pacman.move(dx, dy, maze)

        # 更新幀計數器
        frame_count += 1

        # 移動鬼魂（每隔 GHOST_MOVE_INTERVAL 幀移動一次）
        if frame_count % GHOST_MOVE_INTERVAL == 0:
            for ghost in ghosts:
                ghost.move_random(maze)

        # 檢查 Pac-Man 是否吃到能量球
        pacman.eat_pellet(power_pellets)

        # 檢查 Pac-Man 是否吃到分數球
        pacman.eat_score_pellet(score_pellets)

        # 檢查 Pac-Man 是否與鬼魂碰撞
        if pacman.check_collision(ghosts):
            print(f"Game Over! Score: {pacman.score}")
            running = False

        # 渲染迷宮
        screen.fill(BLACK)
        for y in range(maze.h):
            for x in range(maze.w):
                tile = maze.get_tile(x, y)
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if tile == '#':
                    pygame.draw.rect(screen, BLACK, rect)  # 邊界改為黑色
                elif tile == 'X':
                    pygame.draw.rect(screen, BLACK, rect)  # 牆壁改為黑色
                elif tile == '.':
                    pygame.draw.rect(screen, GRAY, rect)   # 路徑
                elif tile == 'A':
                    pygame.draw.rect(screen, GREEN, rect)  # 死路/偏僻區域
                elif tile == 'S':
                    pygame.draw.rect(screen, GRAY, rect)   # 幽靈生成點（視為路徑）

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
        pacman_rect = pygame.Rect(pacman.x * CELL_SIZE + CELL_SIZE // 4,
                                 pacman.y * CELL_SIZE + CELL_SIZE // 4,
                                 CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.ellipse(screen, YELLOW, pacman_rect)

        # 渲染鬼魂
        for ghost in ghosts:
            ghost_rect = pygame.Rect(ghost.x * CELL_SIZE + CELL_SIZE // 4,
                                    ghost.y * CELL_SIZE + CELL_SIZE // 4,
                                    CELL_SIZE // 2, CELL_SIZE // 2)
            pygame.draw.ellipse(screen, RED, ghost_rect)

        # 渲染分數（白色，立於迷宮之上）
        score_text = font.render(f"Score: {pacman.score}", True, WHITE)
        screen.blit(score_text, (10, 10))  # 在視窗左上角 (10, 10) 位置顯示分數

        # 更新顯示
        pygame.display.flip()
        clock.tick(FPS)

    # 結束遊戲
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()