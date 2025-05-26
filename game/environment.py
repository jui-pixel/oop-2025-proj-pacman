# game/environment.py
"""
定義 Pac-Man 遊戲的強化學習環境，遵循 OpenAI Gym 規範。
負責初始化遊戲迷宮、管理狀態轉換、計算獎勵並提供狀態觀察。
此環境模擬 Pac-Man 遊戲，包含迷宮、Pac-Man、鬼魂、能量球和分數球，
並支持與 DQN 代理的交互。
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
            width (int): 迷宮寬度，預設從 config 導入，用於定義遊戲區域的水平範圍。
            height (int): 迷宮高度，預設從 config 導入，用於定義遊戲區域的垂直範圍。
            seed (int): 隨機種子，用於生成一致的迷宮，確保可重現性，預設從 config 導入。
        """
        # 初始化迷宮生成器，根據給定的寬度和高度創建迷宮結構
        self.maze = Map(w=width, h=height, seed=seed)
        # 生成迷宮，填充牆壁、通道和特殊點（如起點和門）
        self.maze.generate_maze()
        # 初始化遊戲實體，包括 Pac-Man、鬼魂、能量球和分數球
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        # 識別所有重生點（'S' 標記的格子），用於鬼魂重生
        self.respawn_points = [(x, y) for y in range(self.maze.h) for x in range(self.maze.w)
                              if self.maze.get_tile(x, y) == 'S']
        # 跟踪吃掉可食用鬼魂的計分索引，決定獎勵值
        self.ghost_score_index = 0
        # 標記遊戲是否結束，初始為 False
        self.done = False
        # 定義可用的動作空間：0(上), 1(下), 2(左), 3(右)
        self.action_space = [0, 1, 2, 3]
        # 定義觀測空間，形狀為 (高度, 寬度, 通道數)，6 通道分別表示 Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂、牆壁
        self.observation_space = (height, width, 6)
        # 每個格子的像素大小，來自 config，用於渲染
        self.cell_size = CELL_SIZE
        # Pygame 畫面物件，初始為 None，僅在 render 時創建
        self.screen = None
        # Pygame 時鐘物件，控制渲染幀率，初始為 None
        self.clock = None
        # 控制是否啟用渲染，預設為 False
        self.render_enabled = False
        # 記錄 Pac-Man 上一次的位置，用於檢測停滯
        self.last_position = None
        # 計數 Pac-Man 連續停滯的次數
        self.stuck_counter = 0

    def reset(self):
        """
        重置環境，重新生成迷宮和實體，恢復初始狀態。
        這通常在每回合開始時調用，確保環境從一致的起點開始。

        Returns:
            numpy.ndarray: 初始狀態，形狀為 (height, width, 6)，包含所有實體和牆壁的編碼。
        """
        # 重新生成迷宮，確保每次重置都有新佈局
        self.maze.generate_maze()
        # 重新初始化所有遊戲實體，包括位置和狀態
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        # 重置鬼魂計分索引
        self.ghost_score_index = 0
        # 重置遊戲結束標誌
        self.done = False
        # 記錄初始位置，作為停滯檢測的基準
        self.last_position = (self.pacman.x, self.pacman.y)
        # 重置停滯計數器
        self.stuck_counter = 0
        # 返回初始狀態
        return self._get_state()

    def _get_state(self):
        """
        構建當前遊戲狀態，包含 Pac-Man、能量球、分數球、鬼魂和牆壁的二維表示。
        狀態用 6 通道編碼，作為 DQN 代理的輸入。

        Returns:
            numpy.ndarray: 當前狀態，形狀為 (height, width, 6)。
        """
        # 初始化全零狀態陣列，6 通道分別對應不同實體
        state = np.zeros((self.maze.h, self.maze.w, 6), dtype=np.float32)
        # 通道 0：標記 Pac-Man 的位置，值為 1.0
        state[self.pacman.x, self.pacman.y, 0] = 1.0
        # 通道 1：標記能量球位置，值為 1.0
        for pellet in self.power_pellets:
            state[pellet.x, pellet.y, 1] = 1.0
        # 通道 2：標記分數球位置，值為 1.0
        for pellet in self.score_pellets:
            state[pellet.x, pellet.y, 2] = 1.0
        # 通道 3：標記可食用鬼魂位置，值為 1.0（僅在可食用且計時器活躍時）
        # 通道 4：標記普通鬼魂位置，值為 1.0（非可食用狀態）
        for ghost in self.ghosts:
            if ghost.edible and ghost.respawn_timer > 0:
                state[ghost.x, ghost.y, 3] = 1.0
            else:
                state[ghost.x, ghost.y, 4] = 1.0
        # 通道 5：標記牆壁、門和障礙物位置，值為 1.0
        for y in range(self.maze.h):
            for x in range(self.maze.w):
                if self.maze.get_tile(x, y) in ['#', 'X', 'D']:
                    state[x, y, 5] = 1.0
        return state

    def _update_entities(self, action):
        """
        更新 Pac-Man 和鬼魂的狀態，根據動作移動 Pac-Man，鬼魂則追逐或返回重生點。

        Args:
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。
        """
        # 根據動作計算移動方向 (dx, dy)，對應上、下、左、右
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        # 嘗試設置 Pac-Man 的新目標位置，檢查是否可移動
        moving = self.pacman.set_new_target(dx, dy, self.maze)
        # 如果可移動，執行移動
        if moving:
            self.pacman.move_towards_target(self.maze)
        # 更新每個鬼魂的狀態，根據行為邏輯移動
        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                # 如果鬼魂正在返回重生點且到達重生點，設置等待狀態
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) in self.respawn_points:
                    ghost.set_waiting(FPS)
                # 如果正在返回重生點，繼續移動
                elif ghost.returning_to_spawn:
                    ghost.return_to_spawn(self.maze)
                # 否則，根據 Pac-Man 位置移動（追逐行為）
                else:
                    ghost.move(self.pacman, self.maze, FPS)

    def _check_collision(self):
        """
        檢查 Pac-Man 與鬼魂的碰撞，更新分數或結束遊戲。
        使用歐幾里得距離判斷碰撞，處理可食用鬼魂和普通鬼魂的邏輯。

        Returns:
            bool: 是否發生碰撞導致遊戲結束。
        """
        for ghost in self.ghosts:
            # 計算 Pac-Man 和鬼魂之間的歐幾里得距離
            distance = ((self.pacman.x * CELL_SIZE - ghost.x * CELL_SIZE) ** 2 +
                       (self.pacman.y * CELL_SIZE - ghost.y * CELL_SIZE) ** 2) ** 0.5
            # 如果距離小於半個格子，認為發生碰撞
            if distance < CELL_SIZE / 2:
                # 如果鬼魂可食用且計時器活躍，吃掉鬼魂加分
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    ghost.set_returning_to_spawn(FPS)
                # 如果鬼魂不可食用且非返回/等待狀態，遊戲結束
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.done = True
                    return True
        return False

    def _check_stuck(self):
        """
        檢查 Pac-Man 是否停滯（位置未改變）。
        用於檢測代理是否陷入死循環或無法移動。

        Returns:
            bool: 是否停滯（連續停滯次數達到閾值）。
        """
        # 獲取當前 Pac-Man 位置
        current_position = (self.pacman.x, self.pacman.y)
        # 如果位置與上次相同，增加停滯計數
        if self.last_position == current_position:
            self.stuck_counter += 1
        else:
            # 如果位置改變，重置計數
            self.stuck_counter = 0
        # 更新上次位置記錄
        self.last_position = current_position
        # 返回是否停滯，當前閾值為 1（可調整為更高值以減少誤判）
        return self.stuck_counter >= 1  # 立即檢測停滯

    def step(self, action):
        """
        執行一步動作，更新環境狀態並計算獎勵。
        根據動作移動 Pac-Man，更新鬼魂和獎勵，檢查遊戲結束條件。

        Args:
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。

        Returns:
            Tuple: (新狀態, 獎勵, 是否結束, 附加資訊)
                - 新狀態: 新的遊戲狀態陣列。
                - 獎勵: 當前步驟的獎勵值。
                - 是否結束: 遊戲是否結束的布林值。
                - 附加資訊: 目前為空字典。
        """
        # 記錄 Pac-Man 移動前的位置
        old_position = (self.pacman.x, self.pacman.y)
        # 更新所有實體狀態
        self._update_entities(action)

        # 基礎獎勵，鼓勵代理採取動作
        reward = 0.0  # 基礎獎勵
        # # 如果位置改變，額外獎勵移動
        # if (self.pacman.x, self.pacman.y) != old_position:
        #     reward += 0.001  # 獎勵有效移動
        # else:
        #     # 懲罰停滯
        #     reward -= 0.001  # 懲罰停滯
        #     if self._check_stuck():
        #         reward -= 0.01  # 額外懲罰連續停滯

        # 檢查並處理吃掉能量球，增加獎勵並使鬼魂可食用
        if self.pacman.eat_pellet(self.power_pellets) > 0:
            reward += 20  # 獎勵吃掉能量球
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)  # 設置鬼魂可食用狀態

        # 檢查並處理吃掉分數球，增加獎勵
        if self.pacman.eat_score_pellet(self.score_pellets) > 0:
            reward += 2  # 獎勵吃掉分數球

        # 計算與最近鬼魂的距離，鼓勵保持安全距離
        min_ghost_dist = float('inf')
        for ghost in self.ghosts:
            if not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                dist = abs(self.pacman.x - ghost.x) + abs(self.pacman.y - ghost.y)
                min_ghost_dist = min(min_ghost_dist, dist)
        # if min_ghost_dist > 5:
        #     reward += 0.005  # 獎勵與鬼魂保持 5 格以上距離

        # 檢查碰撞，減少獎勵或結束遊戲
        if self._check_collision():
            reward -= 10  # 懲罰碰撞

        # 檢查是否吃完所有球，結束遊戲並給予高獎勵
        if len(self.power_pellets) == 0 and len(self.score_pellets) == 0:
            reward += 10000  # 獎勵完成遊戲
            self.done = True

        return self._get_state(), reward, self.done, {}

    def render(self):
        """
        使用 Pygame 渲染遊戲環境，顯示迷宮和遊戲實體。
        僅在 render_enabled=True 時生效，確保主角清晰可見。
        渲染頻率由 clock.tick(10) 控制，每秒 10 幀。
        """
        if not self.render_enabled:
            # 如果未啟用渲染，直接返回，避免不必要的初始化
            return

        if self.screen is None:
            # 初始化 Pygame 並創建畫面，根據迷宮大小設置解析度
            pygame.init()
            self.screen = pygame.display.set_mode((self.maze.w * self.cell_size, self.maze.h * self.cell_size))
            pygame.display.set_caption("Pac-Man DQN Training")  # 設置視窗標題
            self.clock = pygame.time.Clock()  # 初始化時鐘控制幀率

        for event in pygame.event.get():
            # 處理 Pygame 事件，允許用戶關閉視窗
            if event.type == pygame.QUIT:
                pygame.quit()  # 關閉 Pygame
                return

        # 填充畫面背景為黑色
        self.screen.fill(BLACK)

        # 渲染迷宮格子，根據 tile 類型填充不同顏色
        for y in range(self.maze.h):
            for x in range(self.maze.w):
                tile = self.maze.get_tile(x, y)
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if tile == '#':  # 牆壁
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                elif tile == 'X':  # 通道
                    pygame.draw.rect(self.screen, BLACK, rect)
                elif tile == '.':  # 地板
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif tile == 'E':  # 入口
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif tile == 'S':  # 重生點
                    pygame.draw.rect(self.screen, PINK, rect)
                elif tile == 'D':  # 門
                    pygame.draw.rect(self.screen, RED, rect)

        # 渲染能量球，使用藍色橢圓表示
        for pellet in self.power_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 4,
                pellet.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            pygame.draw.ellipse(self.screen, BLUE, pellet_rect)

        # 渲染分數球，使用橙色橢圓表示
        for pellet in self.score_pellets:
            pellet_rect = pygame.Rect(
                pellet.x * self.cell_size + self.cell_size // 4,
                pellet.y * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2)
            pygame.draw.ellipse(self.screen, ORANGE, pellet_rect)

        # 渲染 Pac-Man，使用黃色橢圓表示
        pacman_rect = pygame.Rect(
            self.pacman.current_x - CELL_SIZE // 4,
            self.pacman.current_y - CELL_SIZE // 4,
            CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.ellipse(self.screen, YELLOW, pacman_rect)

        # 渲染鬼魂，根據狀態使用不同顏色
        for ghost in self.ghosts:
            if ghost.returning_to_spawn:
                base_color = DARK_GRAY  # 返回重生點時為灰色
            elif ghost.edible and ghost.edible_timer > 0:
                base_color = (173, 216, 230)  # 可食用時為淺藍色
                ghost.alpha = 255  # 全透明
            else:
                base_color = ghost.color  # 普通狀態使用鬼魂原色
                ghost.alpha = 255  # 全透明
            # 創建帶透明度的表面
            ghost_surface = pygame.Surface((CELL_SIZE // 2, CELL_SIZE // 2), pygame.SRCALPHA)
            ghost_surface.fill((0, 0, 0, 0))  # 透明背景
            pygame.draw.ellipse(ghost_surface, (*base_color, ghost.alpha),
                               (0, 0, CELL_SIZE // 2, CELL_SIZE // 2))
            # 將鬼魂渲染到畫面
            self.screen.blit(ghost_surface, (ghost.current_x - CELL_SIZE // 4, ghost.current_y - CELL_SIZE // 4))

        # 更新顯示內容
        pygame.display.flip()
        # 控制幀率，每秒 10 幀
        self.clock.tick(10)