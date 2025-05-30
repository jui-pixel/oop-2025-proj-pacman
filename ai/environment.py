# ai/environment.py
"""
定義 Pac-Man 遊戲的強化學習環境，遵循 OpenAI Gym 規範。
負責初始化遊戲迷宮、管理狀態轉換、計算獎勵並提供狀態觀察。
此環境模擬 Pac-Man 遊戲，支援 Dueling DQN 代理的訓練。
"""

import os
import sys
# 修正無效轉義序列警告，並確保正確導入 game 模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) 

import numpy as np
# 確保從正確的路徑導入實體
from game.entities.pacman import PacMan
from game.entities.entity_initializer import initialize_entities
from game.entities.ghost import Ghost1, Ghost2, Ghost3, Ghost4
from game.entities.pellets import PowerPellet, ScorePellet
from game.maze_generator import Map
from config import * # 從 config 導入所有配置參數
import pygame

class PacManEnv:
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        """
        初始化 Pac-Man 遊戲環境，設置迷宮和遊戲實體。

        Args:
            width (int): 迷宮寬度，預設從 config 導入。
            height (int): 迷宮高度，預設從 config 導入。
            seed (int): 隨機種子，預設從 config 導入。
        """
        self.maze = Map(width=width, height=height, seed=seed)
        self.maze.generate_maze()
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        
        self.width = width
        self.height = height
        self.cell_size = CELL_SIZE
        self.seed = seed
        self.current_score = 0
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets) # 初始化總能量球數
        self.eaten_pellets = 0 # 初始化已吃能量球數
        self.game_over = False
        self.ghost_eat_score_multiplier = 1 # 吃鬼魂的分數倍數
        self.edible_timer_start = 0

        # Pygame 相關初始化
        pygame.init()
        # 設置視窗大小，根據迷宮尺寸和單元格大小計算
        self.screen_width = self.width * self.cell_size
        self.screen_height = self.height * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pac-Man RL Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24) # 設置字體用於顯示分數

        self.state_channels = 6 # Pac-Man, PowerPellet, ScorePellet, EdibleGhost, NormalGhost, Wall
        self.state_shape = (self.height, self.width, self.state_channels)


    def _get_state(self):
        """
        生成當前遊戲狀態的 NumPy 陣列表示。
        狀態包含多個通道，每個通道表示一種實體的位置。
        """
        # 初始化所有通道為零
        state = np.zeros((self.height, self.width, self.state_channels), dtype=np.float32)

        # 牆壁通道
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.get_tile(x, y) == '#':
                    state[y, x, 5] = 1.0 # 牆壁在第 5 個通道

        # Pac-Man 位置通道
        state[int(self.pacman.y), int(self.pacman.x), 0] = 1.0

        # 能量球通道
        for pellet in self.power_pellets:
            state[pellet.y, pellet.x, 1] = 1.0

        # 分數球通道
        for pellet in self.score_pellets:
            state[pellet.y, pellet.x, 2] = 1.0

        # 鬼魂通道
        for ghost in self.ghosts:
            if ghost.edible and ghost.edible_timer > 0:
                state[int(ghost.y), int(ghost.x), 3] = 1.0 # 可食用鬼魂在第 3 個通道
            else:
                state[int(ghost.y), int(ghost.x), 4] = 1.0 # 普通鬼魂在第 4 個通道
        
        # 將通道維度從最後一個移到第一個 (channels, height, width)
        # 以符合 PyTorch 的 Conv2d 輸入要求
        state = np.transpose(state, (2, 0, 1)) # (channels, height, width)

        return state


    def reset(self):
        """
        重置遊戲環境到初始狀態。
        """
        self.maze = Map(width=self.width, height=self.height, seed=self.seed)
        self.maze.generate_maze()
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        
        self.current_score = 0
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.ghost_eat_score_multiplier = 1
        self.edible_timer_start = 0
        
        # 重置鬼魂的速度和狀態
        for ghost in self.ghosts:
            ghost.reset_state(self.maze) # 確保鬼魂狀態被完全重置

        return self._get_state()

    def step(self, action):
        """
        執行一個動作並推進遊戲狀態。

        Args:
            action (int): Pac-Man 的動作 (0:上, 1:下, 2:左, 3:右)。

        Returns:
            Tuple[np.array, float, bool, dict]:
                - next_state (np.array): 下一個狀態的觀察。
                - reward (float): 執行動作後的獎勵。
                - done (bool): 遊戲是否結束。
                - info (dict): 額外資訊（例如分數）。
        """
        reward = 0
        info = {}

        # 根據動作更新 Pac-Man 的目標方向
        dx, dy = 0, 0
        if action == 0: dy = -1  # 上
        elif action == 1: dy = 1   # 下
        elif action == 2: dx = -1  # 左
        elif action == 3: dx = 1   # 右
        
        new_pacman_x, new_pacman_y = self.pacman.x + dx, self.pacman.y + dy

        # 檢查移動是否撞牆
        if self.maze.get_tile(int(new_pacman_x), int(new_pacman_y)) == '#':
            reward += -0.1 # 撞牆懲罰
            # Pac-Man 不移動
            self.pacman.update_position(self.maze, 0, 0) # 保持原位
        else:
            # Pac-Man 移動並更新位置
            self.pacman.update_position(self.maze, dx, dy)
            reward += -0.01 # 每移動一步的微小時間懲罰

            # 檢查是否吃到能量球
            for pellet in list(self.power_pellets): # 遍歷副本以安全移除
                if pellet.x == self.pacman.x and pellet.y == self.pacman.y:
                    self.power_pellets.remove(pellet)
                    reward += 50.0 # 吃到能量球獎勵
                    self.current_score += 50
                    self.eaten_pellets += 1
                    self.ghost_eat_score_multiplier = 1 # 重置倍數
                    # 讓所有鬼魂進入可食用狀態
                    for ghost in self.ghosts:
                        ghost.set_edible(EDIBLE_DURATION)
                    self.edible_timer_start = pygame.time.get_ticks()
                    break

            # 檢查是否吃到分數球
            for pellet in list(self.score_pellets): # 遍歷副本以安全移除
                if pellet.x == self.pacman.x and pellet.y == self.pacman.y:
                    self.score_pellets.remove(pellet)
                    reward += 10.0 # 吃到分數球獎勵
                    self.current_score += 10
                    self.eaten_pellets += 1
                    break
        
        # 更新鬼魂狀態和位置
        for ghost in self.ghosts:
            ghost.update_target(self.pacman, self.maze, self.ghosts) # 根據 Pac-Man 位置更新目標
            ghost.move(self.maze) # 鬼魂移動
            # 更新可食用計時器
            if ghost.edible and ghost.edible_timer > 0:
                time_passed = (pygame.time.get_ticks() - self.edible_timer_start) / 1000.0
                ghost.edible_timer = EDIBLE_DURATION - time_passed
                if ghost.edible_timer <= 0:
                    ghost.edible = False # 計時器結束，不再可食用

        # 碰撞檢測 (Pac-Man 和鬼魂)
        for ghost in self.ghosts:
            if int(self.pacman.x) == int(ghost.x) and int(self.pacman.y) == int(ghost.y):
                if ghost.edible and ghost.edible_timer > 0 and not ghost.returning_to_spawn:
                    # 吃到可食用鬼魂
                    reward += GHOST_SCORES[self.ghost_eat_score_multiplier] # 根據倍數給予獎勵
                    self.current_score += GHOST_SCORES[self.ghost_eat_score_multiplier]
                    self.ghost_eat_score_multiplier *= 2 # 倍數翻倍
                    ghost.returning_to_spawn = True # 鬼魂返回重生點
                    ghost.edible = False # 鬼魂不再可食用
                elif not ghost.edible and not ghost.returning_to_spawn:
                    # 被普通鬼魂抓住，遊戲結束
                    reward += -100.0 # 被抓懲罰
                    self.game_over = True
                    break # 遊戲結束，跳出循環

        # 遊戲勝利條件：所有能量球和分數球被吃完
        if not self.power_pellets and not self.score_pellets:
            reward += 200.0 # 遊戲勝利獎勵
            self.game_over = True

        next_state = self._get_state()
        info['score'] = self.current_score
        info['game_over'] = self.game_over
        info['eaten_pellets'] = self.eaten_pellets
        info['total_pellets'] = self.total_pellets

        return next_state, reward, self.game_over, info

    def render(self):
        """
        使用 Pygame 渲染遊戲畫面。
        """
        self.screen.fill(BLACK) # 清空屏幕

        # 繪製迷宮牆壁
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.get_tile(x, y) == '#':
                    pygame.draw.rect(self.screen, BLUE, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
                elif self.maze.get_tile(x, y) == 'X': # 繪製中央房間的牆壁
                    pygame.draw.rect(self.screen, GRAY, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
                elif self.maze.get_tile(x, y) == 'D': # 繪製門
                    pygame.draw.rect(self.screen, DARK_GRAY, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # 繪製能量球
        for pellet in self.power_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 4,
                pellet.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            pygame.draw.ellipse(self.screen, GREEN, pellet_rect) # 能量球改用綠色

        # 繪製分數球
        for pellet in self.score_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 3, # 稍微縮小以區分
                pellet.y * self.cell_size + self.cell_size // 3,
                self.cell_size // 3, self.cell_size // 3)
            pygame.draw.ellipse(self.screen, ORANGE, pellet_rect)

        # 繪製 Pac-Man
        pacman_rect = pygame.Rect(
            self.pacman.current_x - self.cell_size // 4,
            self.pacman.current_y - self.cell_size // 4,
            self.cell_size // 2, self.cell_size // 2)
        pygame.draw.ellipse(self.screen, YELLOW, pacman_rect)

        # 繪製鬼魂
        for ghost in self.ghosts:
            base_color = ghost.color
            if ghost.returning_to_spawn:
                base_color = DARK_GRAY # 返回重生點的鬼魂變灰
            elif ghost.edible and ghost.edible_timer > 0:
                # 可食用鬼魂閃爍效果
                flash_interval = 200 # 毫秒
                if ghost.edible_timer * 1000 < 3000 and (pygame.time.get_ticks() // flash_interval) % 2 == 0:
                     base_color = (173, 216, 230) # 淺藍色 (可食用顏色)
                else:
                    base_color = (0, 0, 255) # 深藍色 (可食用顏色)

            ghost_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA) # 使用整個 cell_size
            ghost_rect = pygame.Rect(
                ghost.current_x - self.cell_size // 2,
                ghost.current_y - self.cell_size // 2,
                self.cell_size, self.cell_size # 使用整個 cell_size
            )
            pygame.draw.circle(ghost_surface, base_color, (self.cell_size // 2, self.cell_size // 2), self.cell_size // 3)
            # 繪製鬼魂底部
            pygame.draw.rect(ghost_surface, base_color, (0, self.cell_size // 2, self.cell_size, self.cell_size // 2))
            
            # 繪製鬼魂眼睛
            eye_radius = self.cell_size // 10
            # 左眼
            pygame.draw.circle(ghost_surface, WHITE, (self.cell_size // 2 - self.cell_size // 8, self.cell_size // 2 - self.cell_size // 10), eye_radius)
            pygame.draw.circle(ghost_surface, BLACK, (self.cell_size // 2 - self.cell_size // 8, self.cell_size // 2 - self.cell_size // 10), eye_radius // 2)
            # 右眼
            pygame.draw.circle(ghost_surface, WHITE, (self.cell_size // 2 + self.cell_size // 8, self.cell_size // 2 - self.cell_size // 10), eye_radius)
            pygame.draw.circle(ghost_surface, BLACK, (self.cell_size // 2 + self.cell_size // 8, self.cell_size // 2 - self.cell_size // 10), eye_radius // 2)

            self.screen.blit(ghost_surface, ghost_rect)

        # 顯示當前分數
        score_text = self.font.render(f"Score: {self.current_score}", True, WHITE)
        self.screen.blit(score_text, (5, 5))

        pygame.display.flip() # 更新顯示

    def close(self):
        """
        關閉 Pygame 視窗。
        """
        pygame.quit()