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
        self.ghost_score_index = 0
        self.done = False
        self.action_space = [0, 1, 2, 3]
        self.observation_space = (height, width, 6)
        self.cell_size = CELL_SIZE
        self.screen = None
        self.clock = None
        self.render_enabled = False
        self.last_position = None
        self.stuck_counter = 0

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
        self.last_position = (self.pacman.x, self.pacman.y)
        self.stuck_counter = 0
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
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        moving = self.pacman.set_new_target(dx, dy, self.maze)
        if moving:
            self.pacman.move_towards_target(self.maze)

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
                    ghost.return_to_spawn(maze=self.maze)
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.done = True
                    return True
        return False

    def _check_stuck(self):
        """
        檢查 Pac-Man 是否停滯（位置未改變）。

        Returns:
            bool: 是否停滯。
        """
        current_position = (self.pacman.x, self.pacman.y)
        if self.last_position == current_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = current_position
        return self.stuck_counter >= 1  # 立即檢測停滯

    def step(self, action):
        """
        執行一步動作，更新環境狀態並計算獎勵。

        Args:
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。

        Returns:
            Tuple: (新狀態, 獎勵, 是否結束, 附加資訊)
        """
        old_position = (self.pacman.x, self.pacman.y)
        self._update_entities(action)

        reward = 0.001  # 獎勵為 0.001
        if (self.pacman.x, self.pacman.y) != old_position:
            reward += 0.003  # 獎勵有效移動
        else:
            reward -= 0.001  # 懲罰停滯
            if self._check_stuck():
                reward -= 0.01  # 額外懲罰連續停滯

        if self.pacman.eat_pellet(self.power_pellets) > 0:
            reward += 20
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)
        if self.pacman.eat_score_pellet(self.score_pellets) > 0:
            reward += 2

        min_ghost_dist = float('inf')
        for ghost in self.ghosts:
            if not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                dist = abs(self.pacman.x - ghost.x) + abs(self.pacman.y - ghost.y)
                min_ghost_dist = min(min_ghost_dist, dist)
        if min_ghost_dist > 5:
            reward += 0.005

        if self._check_collision():
            reward -= 100

        if len(self.power_pellets) == 0 and len(self.score_pellets) == 0:
            reward += 10000
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

        self.screen.fill(BLACK)

        for y in range(self.maze.h):
            for x in range(self.maze.w):
                tile = self.maze.get_tile(x, y)
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if tile == '#':
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                elif tile == 'X':
                    pygame.draw.rect(self.screen, BLACK, rect)
                elif tile == '.':
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif tile == 'E':
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif tile == 'S':
                    pygame.draw.rect(self.screen, PINK, rect)
                elif tile == 'D':
                    pygame.draw.rect(self.screen, RED, rect)

        for pellet in self.power_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 4,
                pellet.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            pygame.draw.ellipse(self.screen, BLUE, pellet_rect)

        for pellet in self.score_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 4,
                pellet.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            pygame.draw.ellipse(self.screen, ORANGE, pellet_rect)

        pacman_rect = pygame.Rect(
            self.pacman.current_x - CELL_SIZE // 4,
            self.pacman.current_y - CELL_SIZE // 4,
            CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.ellipse(self.screen, YELLOW, pacman_rect)

        for ghost in self.ghosts:
            if ghost.returning_to_spawn:
                base_color = DARK_GRAY
            elif ghost.edible and ghost.edible_timer > 0:
                base_color = (173, 216, 230)
                ghost.alpha = 255
            else:
                base_color = ghost.color
                ghost.alpha = 255
            
            ghost_surface = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2), pygame.SRCALPHA)
            ghost_surface.fill((0, 0, 0, 0))  # 透明背景
            pygame.draw.ellipse(ghost_surface, (*base_color, ghost.alpha),
                               (0, 0, CELL_SIZE // 2, CELL_SIZE // 2))
            self.screen.blit(ghost_surface, (ghost.current_x - CELL_SIZE // 4, ghost.current_y - CELL_SIZE // 4))

        pygame.display.flip()
        self.clock.tick(10)