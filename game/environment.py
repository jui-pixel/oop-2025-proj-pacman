# game/environment.py
"""
定義 Pac-Man 遊戲的強化學習環境，遵循 OpenAI Gym 規範。
負責初始化遊戲迷宮、管理狀態轉換、計算獎勵並提供狀態觀察。
"""

import numpy as np
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, EDIBLE_DURATION, GHOST_SCORES, FPS, CELL_SIZE, BLACK, DARK_GRAY, GRAY, GREEN, PINK, RED, BLUE, ORANGE, YELLOW
import pygame

class PacManEnv:
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        """
        初始化 Pac-Man 遊戲環境，設置迷宮和遊戲實體。

        Args:
            width (int): 迷宮寬度，預設從 config 導入。
            height (int): 迷宮高度，預設從 config 導入。
            seed (int): 隨機種子，用於生成一致的迷宮，預設從 config 導入。
        """
        self.maze = Map(w=width, h=height, seed=seed)
        self.maze.generate_maze()
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.respawn_points = [(x, y) for y in range(self.maze.h) for x in range(self.maze.w)
                              if self.maze.get_tile(x, y) == 'S']
        self.ghost_score_index = 0  # 鬼魂分數索引
        self.done = False
        self.action_space = [0, 1, 2, 3]  # 動作：上、下、左、右
        self.observation_space = (height, width, 6)
        self.cell_size = CELL_SIZE  # 使用 config 中的 CELL_SIZE
        self.screen = None
        self.clock = None
        self.render_enabled = False

    def reset(self):
        """
        重置環境，重新生成迷宮和實體，恢復初始狀態。

        Returns:
            numpy.ndarray: 初始狀態，形狀為 (height, width, 6)。
        """
        self.maze.generate_maze()
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.ghost_score_index = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        構建當前遊戲狀態，包含 Pac-Man、能量球、分數球、鬼魂和牆壁的二維表示。

        Returns:
            numpy.ndarray: 當前狀態，形狀為 (height, width, 6)。
        """
        state = np.zeros((self.maze.h, self.maze.w, 6), dtype=np.float32)
        state[self.pacman.x, self.pacman.y, 0] = 1.0
        for pellet in self.power_pellets:
            state[pellet.x, pellet.y, 1] = 1.0
        for pellet in self.score_pellets:
            state[pellet.x, pellet.y, 2] = 1.0
        for ghost in self.ghosts:
            if ghost.edible and ghost.respawn_timer > 0:
                state[ghost.x, ghost.y, 3] = 1.0
            else:
                state[ghost.x, ghost.y, 4] = 1.0
        for y in range(self.maze.h):
            for x in range(self.maze.w):
                if self.maze.get_tile(x, y) in ['#', 'X', 'D']:
                    state[x, y, 5] = 1.0
        return state

    def _update_entities(self, action):
        """
        更新 Pac-Man 和鬼魂的狀態。

        Args:
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。
        """
        # 移動 Pac-Man
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        moving = self.pacman.set_new_target(dx, dy, self.maze)
        if moving:
            self.pacman.move_towards_target(self.maze)

        # 移動鬼魂
        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) in self.respawn_points:
                    ghost.set_waiting(FPS)
                else:
                    ghost.move(self.pacman, self.maze, FPS)

    def _check_collision(self):
        """
        檢查 Pac-Man 與鬼魂的碰撞，更新分數或結束遊戲。
        """
        for ghost in self.ghosts:
            distance = ((self.pacman.x * CELL_SIZE - ghost.x * CELL_SIZE) ** 2 +
                       (self.pacman.y * CELL_SIZE - ghost.y * CELL_SIZE) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    ghost.set_returning_to_spawn(FPS)
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.done = True
                break

    def step(self, action):
        """
        執行一步動作，更新環境狀態並計算獎勵。

        Args:
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。

        Returns:
            Tuple: (新狀態, 獎勵, 是否結束, 附加資訊)
        """
        # 更新實體狀態
        self._update_entities(action)

        # 計算獎勵
        reward = -0.1  # 基本懲罰，鼓勵快速行動
        if self.pacman.eat_pellet(self.power_pellets) > 0:
            reward = 40  # 吃到能量球的獎勵
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)  # 設置鬼魂為可吃
        if self.pacman.eat_score_pellet(self.score_pellets) > 0:
            reward = 10  # 吃到分數球的獎勵

        # 檢查鬼魂距離獎勵
        min_ghost_dist = float('inf')
        for ghost in self.ghosts:
            if not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                dist = abs(self.pacman.x - ghost.x) + abs(self.pacman.y - ghost.y)
                min_ghost_dist = min(min_ghost_dist, dist)
        if min_ghost_dist > 5:
            reward += 0.5  # 與鬼魂保持距離的獎勵

        # 檢查碰撞
        self._check_collision()

        # 檢查遊戲結束條件
        if len(self.power_pellets) == 0 and len(self.score_pellets) == 0:
            reward += 1000  # 完成遊戲的獎勵
            self.done = True

        return self._get_state(), reward, self.done, {}

    def render(self):
        """
        使用 Pygame 渲染遊戲環境，顯示迷宮和遊戲實體。
        僅在 render_enabled=True 時生效，確保主角清晰可見。
        """
        if not self.render_enabled:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.maze.w * self.cell_size, self.maze.h * self.cell_size))
            pygame.display.set_caption("Pac-Man DQN Training")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill(BLACK)  # 黑色背景

        # 渲染迷宮
        for y in range(self.maze.h):
            for x in range(self.maze.w):
                tile = self.maze.get_tile(x, y)
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if tile == '#':
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)  # 邊界
                elif tile == 'X':
                    pygame.draw.rect(self.screen, BLACK, rect)  # 牆壁
                elif tile == '.':
                    pygame.draw.rect(self.screen, GRAY, rect)  # 路徑
                elif tile == 'E':
                    pygame.draw.rect(self.screen, GREEN, rect)  # 能量球位置
                elif tile == 'S':
                    pygame.draw.rect(self.screen, PINK, rect)  # 重生點
                elif tile == 'D':
                    pygame.draw.rect(self.screen, RED, rect)  # 門

        # 渲染能量球
        for pellet in self.power_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 4,
                pellet.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            pygame.draw.ellipse(self.screen, BLUE, pellet_rect)

        # 渲染分數球
        for pellet in self.score_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 4,
                pellet.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            pygame.draw.ellipse(self.screen, ORANGE, pellet_rect)

        # 渲染 Pac-Man
        pacman_rect = pygame.Rect(
            self.pacman.x * self.cell_size + self.cell_size // 4,
            self.pacman.y * self.cell_size + self.cell_size // 4,
            self.cell_size // 2, self.cell_size // 2)
        pygame.draw.ellipse(self.screen, YELLOW, pacman_rect)

        # 渲染鬼魂
        for ghost in self.ghosts:
            ghost_rect = pygame.Rect(
                ghost.x * self.cell_size + self.cell_size // 4,
                ghost.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            if ghost.returning_to_spawn:
                pygame.draw.ellipse(self.screen, DARK_GRAY, ghost_rect)  # 固定灰色
            elif ghost.edible and ghost.respawn_timer > 0:
                pygame.draw.ellipse(self.screen, (173, 216, 230), ghost_rect)  # 固定淺藍色
            else:
                pygame.draw.ellipse(self.screen, ghost.color, ghost_rect)  # 固定鬼魂顏色

        pygame.display.flip()
        self.clock.tick(30)