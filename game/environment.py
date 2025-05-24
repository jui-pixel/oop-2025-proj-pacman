# game/environment.py
"""
定義 Pac-Man 遊戲的強化學習環境，遵循 OpenAI Gym 規範。
負責初始化遊戲迷宮、管理狀態轉換、計算獎勵並提供狀態觀察。
"""

import numpy as np
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED
import pygame  # 引入 Pygame 用於可視化

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
        self.done = False
        self.action_space = [0, 1, 2, 3]
        self.observation_space = (height, width, 6)
        # Pygame 可視化相關
        self.cell_size = 20  # 每個格子的大小（像素）
        self.screen = None  # Pygame 視窗
        self.clock = None  # 控制幀率
        self.render_enabled = False  # 可視化是否啟用

    def reset(self):
        """
        重置環境，重新生成迷宮和實體，恢復初始狀態。

        Returns:
            numpy.ndarray: 初始狀態，形狀為 (height, width, 6)。
        """
        self.maze.generate_maze()
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
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

    def step(self, action):
        """
        執行一步動作，更新環境狀態並計算獎勵。

        Args:
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。

        Returns:
            Tuple: (新狀態, 獎勵, 是否結束, 附加資訊)
        """
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        moving = self.pacman.set_new_target(dx, dy, self.maze)
        if moving:
            self.pacman.move_towards_target(self.maze)

        reward = -0.1
        if self.pacman.eat_pellet(self.power_pellets) > 0:
            reward = 40
        if self.pacman.eat_score_pellet(self.score_pellets) > 0:
            reward = 10

        min_ghost_dist = float('inf')
        for ghost in self.ghosts:
            if not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                dist = abs(self.pacman.x - ghost.x) + abs(self.pacman.y - ghost.y)
                min_ghost_dist = min(min_ghost_dist, dist)
        if min_ghost_dist > 5:
            reward += 0.5

        for ghost in self.ghosts:
            if self.pacman.x == ghost.x and self.pacman.y == ghost.y:
                if ghost.edible and ghost.respawn_timer > 0:
                    reward = [100, 150, 250, 400][min(ghost.death_count, 3)]
                    ghost.set_returning_to_spawn(30)
                elif not ghost.returning_to_spawn and not ghost.waiting:
                    reward = -100
                    self.done = True

        if len(self.power_pellets) == 0 and len(self.score_pellets) == 0:
            reward += 1000
            self.done = True

        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == 'S':
                    ghost.set_waiting(30)
                else:
                    ghost.move(self.pacman, self.maze, 30)

        return self._get_state(), reward, self.done, {}

    def render(self):
        """
        使用 Pygame 渲染遊戲環境，顯示迷宮和遊戲實體。
        僅在 render_enabled=True 時生效。
        """
        if not self.render_enabled:
            return

        # 初始化 Pygame（僅在第一次調用 render 時）
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.maze.w * self.cell_size, self.maze.h * self.cell_size))
            pygame.display.set_caption("Pac-Man DQN Training")
            self.clock = pygame.time.Clock()

        # 檢查 Pygame 事件（允許關閉視窗）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 清空畫面
        self.screen.fill((0, 0, 0))  # 黑色背景

        # 繪製迷宮
        for y in range(self.maze.h):
            for x in range(self.maze.w):
                tile = self.maze.get_tile(x, y)
                if tile in ['#', 'X', 'D']:
                    pygame.draw.rect(self.screen, (0, 0, 255),  # 藍色牆壁
                                     (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
                elif tile == 'S':
                    pygame.draw.rect(self.screen, (255, 165, 0),  # 橙色重生點
                                     (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # 繪製分數球
        for pellet in self.score_pellets:
            pygame.draw.circle(self.screen, (255, 255, 255),  # 白色分數球
                               (pellet.x * self.cell_size + self.cell_size // 2, pellet.y * self.cell_size + self.cell_size // 2),
                               self.cell_size // 4)

        # 繪製能量球
        for pellet in self.power_pellets:
            pygame.draw.circle(self.screen, (255, 255, 0),  # 黃色能量球
                               (pellet.x * self.cell_size + self.cell_size // 2, pellet.y * self.cell_size + self.cell_size // 2),
                               self.cell_size // 2)

        # 繪製鬼魂
        for ghost in self.ghosts:
            color = (0, 0, 255) if ghost.edible else (255, 0, 0)  # 可吃時藍色，否則紅色
            pygame.draw.rect(self.screen, color,
                             (ghost.x * self.cell_size, ghost.y * self.cell_size, self.cell_size, self.cell_size))

        # 繪製 Pac-Man
        pygame.draw.circle(self.screen, (255, 255, 0),  # 黃色 Pac-Man
                           (self.pacman.x * self.cell_size + self.cell_size // 2, self.pacman.y * self.cell_size + self.cell_size // 2),
                           self.cell_size // 2)

        # 更新畫面並控制幀率
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS