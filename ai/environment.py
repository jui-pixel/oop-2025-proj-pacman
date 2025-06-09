# ai/environment.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 將父目錄加入路徑以便導入其他模塊
import numpy as np
from gym.spaces import Discrete, Box
from game.game import Game
from config import *  # 導入配置參數（如迷宮尺寸、格子大小等）
from typing import Callable
import random

class PacManEnv(Game):
    metadata = {"render_modes": [], "render_fps": None}  # 元數據，暫無渲染功能

    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        """
        初始化 Pac-Man 環境，基於 Game 類，提供強化學習的標準接口。

        原理：
        - 本環境模擬經典 Pac-Man 遊戲，作為強化學習的測試平台，遵循 OpenAI Gym 的接口規範。
        - 環境狀態以 6 通道的張量表示，涵蓋 Pac-Man 位置、能量球、分數球、可食用鬼魂、普通鬼魂和牆壁。
        - 動作空間為離散的 4 個方向（上、下、左、右），獎勵基於分數增量和碰撞懲罰。
        - 環境支持隨機種子和隨機出生點，增加訓練數據的多樣性。
        - 隨機種子用於確保可重現性，影響迷宮生成和初始位置。

        Args:
            width (int): 迷宮寬度，預設從 config 載入（MAZE_WIDTH）。
            height (int): 迷宮高度，預設從 config 載入（MAZE_HEIGHT）。
            seed (int): 隨機種子，預設從 config 載入（MAZE_SEED），確保可重現性。
        """
        super().__init__(player_name="RL_Agent")  # 調用父類 Game 的初始化，設置玩家名稱
        self.width = width  # 迷宮寬度
        self.height = height  # 迷宮高度
        self.cell_size = CELL_SIZE  # 每個格子的像素大小（來自 config）
        self.seed = seed  # 隨機種子
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)  # 總彈丸數（分數球 + 能量球）
        self.eaten_pellets = 0  # 已吃彈丸數
        self.game_over = False  # 遊戲是否結束
        self.current_score = 0  # 當前分數
        self.old_score = 0  # 上一次分數，用於計算獎勵
        self.frame_count = 0  # 幀數計數，追蹤遊戲步數
        self.ghost_move_counter = 2  # 鬼魂移動頻率（每 2 幀移動一次）
        self.state_channels = 6  # 狀態通道數（6 個特徵圖）
        self.state_shape = (self.state_channels, self.height, self.width)  # 狀態張量形狀

        # 定義動作空間：離散的 4 個動作（0=上, 1=下, 2=左, 3=右）
        self.action_space = Discrete(4)
        # 定義觀察空間：6 通道的浮點數張量，值範圍 [0, 1]
        self.observation_space = Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)

        np.random.seed(seed)  # 設置 NumPy 隨機種子
        print(f"初始化 PacManEnv：寬度={width}，高度={height}，種子={seed}，生命數={self.pacman.lives}")

    def _get_state(self):
        """
        獲取當前遊戲狀態，返回 6 通道的張量表示。

        原理：
        - 狀態以 6 通道的張量表示，每個通道為二值圖（0 或 1），形狀為 (6, height, width)。
        - 各通道含義：
          - 通道 0：Pac-Man 位置（1 表示存在，0 表示無）。
          - 通道 1：能量球位置。
          - 通道 2：分數球位置。
          - 通道 3：可食用鬼魂位置（僅當鬼魂可食用且未返回重生點）。
          - 通道 4：普通鬼魂位置。
          - 通道 5：牆壁和邊界位置。
        - 這種表示方式將遊戲的空間信息轉化為結構化輸入，適合卷積神經網絡處理。

        Returns:
            np.ndarray: 狀態張量，形狀為 (6, height, width)，數據類型為 float32。
        """
        state = np.zeros((self.state_channels, self.height, self.width), dtype=np.float32)
        # 設置牆壁和邊界
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.get_tile(x, y) in ['#', 'X']:  # '#' 表示牆壁，'X' 表示邊界
                    state[5, y, x] = 1.0
        # 設置 Pac-Man 位置
        state[0, self.pacman.y, self.pacman.x] = 1.0
        # 設置能量球位置
        for pellet in self.power_pellets:
            state[1, pellet.y, pellet.x] = 1.0
        # 設置分數球位置
        for pellet in self.score_pellets:
            state[2, pellet.y, pellet.x] = 1.0
        # 設置鬼魂位置
        for ghost in self.ghosts:
            if ghost.edible and ghost.edible_timer > 3 and not ghost.returning_to_spawn:
                state[3, ghost.y, ghost.x] = 1.0  # 可食用鬼魂
            else:
                state[4, ghost.y, ghost.x] = 1.0  # 普通鬼魂
        return state

    def get_expert_action(self):
        """
        使用規則基礎 AI 獲取專家動作，作為模仿學習的參考。

        原理：
        - 專家動作由 Pac-Man 的規則基礎 AI（rule_based_ai_move）生成，通常基於路徑規劃或啟發式策略。
        - 如果 AI 移動成功，返回對應的動作索引（0=上, 1=下, 2=左, 3=右）。
        - 如果移動失敗（例如無效方向），則從合法動作中隨機選擇，確保動作有效性。
        - 合法動作需滿足：
          - 目標位置在迷宮內（xy_valid）。
          - 目標位置不是牆壁、邊界、門或鬼魂重生點。

        Returns:
            int: 專家選擇的動作索引（0=上, 1=下, 2=左, 3=右）。
        """
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 對應上、下、左、右
        # 調用 Pac-Man 的規則基礎 AI 移動
        success = self.pacman.rule_based_ai_move(self.maze, self.power_pellets, self.score_pellets, self.ghosts)
        if success and self.pacman.last_direction:
            dx, dy = self.pacman.last_direction
            # 將方向轉換為動作索引
            for i, (dx_dir, dy_dir) in enumerate(directions):
                if dx == dx_dir and dy == dy_dir:
                    return i
        # 如果 AI 移動失敗，隨機選擇合法動作
        safe_directions = [i for i, (dx, dy) in enumerate(directions) 
                          if self.maze.xy_valid(self.pacman.x + dx, self.pacman.y + dy)
                          and self.maze.get_tile(self.pacman.x + dx, self.pacman.y + dy) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
        return random.choice(safe_directions) if safe_directions else 0  # 若無合法動作，返回 0（上）

    def reset(self, seed=MAZE_SEED, random_spawn_seed=0):
        """
        重置環境，重新開始遊戲，設置初始狀態。

        原理：
        - 重置環境以開始新回合，重新初始化迷宮、Pac-Man 和鬼魂。
        - 支持隨機出生點（random_spawn_seed），增加訓練數據的多樣性。
        - 隨機種子用於控制迷宮生成和初始位置，確保可重現性。
        - 若提供 random_spawn_seed，Pac-Man 將在有效路徑位置隨機出生。

        Args:
            seed (int, optional): 隨機種子，若提供則更新。
            random_spawn_seed (int, optional): 隨機出生點種子，預設為 0（不隨機）。

        Returns:
            tuple: (狀態張量, 資訊字典)，狀態張量為當前遊戲狀態，資訊字典包含元數據。
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        super().__init__(player_name="RL_Agent")  # 重置遊戲
        if random_spawn_seed != 0:
            random.seed(self.seed + random_spawn_seed)  # 使用偏移種子
            # 選擇有效路徑位置作為出生點
            valid_positions = [(x, y) for y in range(1, self.maze.height - 1) for x in range(1, self.maze.width - 1)
                              if self.maze.get_tile(x, y) == TILE_PATH]
            self.pacman.x, self.pacman.y = random.choice(valid_positions)
            self.pacman.initial_x = self.pacman.x
            self.pacman.initial_y = self.pacman.y
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)  # 重置總彈丸數
        self.eaten_pellets = 0  # 重置已吃彈丸數
        self.game_over = False  # 重置遊戲結束標誌
        self.current_score = 0  # 重置當前分數
        self.old_score = 0  # 重置上次分數
        self.frame_count = 0  # 重置幀數
        self.ghost_move_counter = 2  # 重置鬼魂移動計數器
        state = self._get_state()  # 獲取初始狀態
        return np.array(state, dtype=np.float32), {}

    def update(self, fps: int, move_pacman: Callable[[], None]) -> None:
        """
        更新遊戲狀態，包括 Pac-Man 和鬼魂的移動、碰撞檢測等。

        原理：
        - 每步更新執行以下邏輯：
          1. 移動 Pac-Man（由外部提供的 move_pacman 函數控制）。
          2. 檢查是否吃到能量球或分數球，更新分數和鬼魂狀態。
          3. 每隔 ghost_move_counter 幀移動鬼魂，根據其狀態（正常、可食用、返回重生點）執行不同邏輯。
          4. 檢查 Pac-Man 與鬼魂的碰撞，根據鬼魂狀態計算分數或損失生命。
          5. 檢查是否吃完所有彈丸，結束遊戲。
        - 鬼魂移動頻率低於 Pac-Man（每 2 幀一次），模擬遊戲的動態平衡。

        Args:
            fps (int): 每秒幀數，用於時間計算（來自 config 的 FPS）。
            move_pacman (Callable[[], None]): 控制 Pac-Man 移動的函數。
        """
        move_pacman()  # 執行 Pac-Man 移動

        # 檢查是否吃到能量球
        score_from_pellet = self.pacman.eat_pellet(self.power_pellets)
        if score_from_pellet > 0:
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)  # 設置鬼魂為可食用狀態，持續 EDIBLE_DURATION 幀

        # 檢查是否吃到分數球
        self.pacman.eat_score_pellet(self.score_pellets)

        # 移動所有鬼魂
        for ghost in self.ghosts:
            if ghost.move_towards_target(FPS):  # 僅在到達目標時更新
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == TILE_GHOST_SPAWN:
                    ghost.set_waiting(FPS)  # 鬼魂到達重生點後等待
                elif ghost.returning_to_spawn:
                    ghost.return_to_spawn(self.maze)  # 繼續返回重生點
                else:
                    if self.frame_count % self.ghost_move_counter == 0:
                        ghost.move(self.pacman, self.maze, FPS)  # 執行鬼魂移動邏輯
                        # 更新鬼魂的像素坐標
                        ghost.current_x = ghost.target_x * CELL_SIZE + CELL_SIZE // 2
                        ghost.current_y = ghost.target_y * CELL_SIZE + CELL_SIZE // 2
                        ghost.x = ghost.target_x
                        ghost.y = ghost.target_y

        # 檢查碰撞
        self._check_collision(FPS)

        # 檢查是否吃完所有彈丸
        if not self.power_pellets and not self.score_pellets:
            print(f"遊戲勝利！所有彈丸已收集。最終分數：{self.pacman.score}")
            self.running = False  # 設置遊戲結束
            self.game_over = True

    def _check_collision(self, fps: int) -> None:
        """
        檢查 Pac-Man 與鬼魂的碰撞，更新分數或觸發死亡。

        原理：
        - 碰撞檢測基於 Pac-Man 和鬼魂的像素坐標距離，公式：distance = √((x_p - x_g)^2 + (y_p - y_g)^2)。
        - 若距離小於格子大小的一半（CELL_SIZE / 2），則認為發生碰撞。
        - 碰撞後果：
          - 若鬼魂可食用（edible），增加分數（GHOST_SCORES），鬼魂返回重生點。
          - 若鬼魂不可食用，Pac-Man 損失生命，鬼魂返回重生點。
          - 若 Pac-Man 生命耗盡，遊戲結束。

        Args:
            fps (int): 每秒幀數，用於時間計算。
        """
        if not self.running:
            return
        
        for ghost in self.ghosts:
            # 計算 Pac-Man 與鬼魂的歐幾里得距離
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:  # 碰撞檢測
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]  # 增加分數
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)  # 更新分數索引
                    ghost.set_returning_to_spawn(FPS)  # 鬼魂返回重生點
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.pacman.lose_life(self.maze)  # Pac-Man 損失生命
                    for g in self.ghosts:  # 所有鬼魂返回重生點
                        g.set_returning_to_spawn(FPS)
                    if self.pacman.lives <= 0:
                        self.running = False  # 遊戲結束
                        self.game_over = True
                    else:
                        break

    def step(self, action):
        """
        執行一步動作，更新遊戲狀態並返回結果。

        原理：
        - 根據輸入動作（上、下、左、右）更新 Pac-Man 位置，檢查碰撞和獎勵。
        - 獎勵計算：reward = current_score - old_score - wall_penalty
          - current_score：當前分數（來自吃彈丸或鬼魂）。
          - old_score：上一步分數。
          - wall_penalty：撞牆懲罰（-1）。
        - 有效步驟（valid_step）表示 Pac-Man 成功移動且未終止。
        - 環境返回標準五元組：(下一狀態, 獎勵, 是否終止, 是否截斷, 資訊字典)。

        Args:
            action (int): 動作索引（0=上, 1=下, 2=左, 3=右）。

        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
            - next_state: 下一狀態張量。
            - reward: 本步獎勵。
            - terminated: 是否因遊戲結束（勝利或失敗）終止。
            - truncated: 是否因其他原因截斷（此處與 terminated 相同）。
            - info: 包含幀數、分數、彈丸數等信息的字典。
        """
        if not 0 <= action < 4:
            raise ValueError(f"無效動作：{action}")

        moved = True  # 是否成功移動
        wall_collision = False  # 是否撞牆
        
        def move_pacman():
            nonlocal moved, wall_collision
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上、下、左、右
            dx, dy = directions[action]
            if self.pacman.move_towards_target(FPS):  # 僅在到達目標格子時更新
                if not self.pacman.set_new_target(dx, dy, self.maze):
                    wall_collision = True  # 撞牆
                else:
                    # 更新 Pac-Man 的像素坐標
                    self.pacman.current_x = self.pacman.target_x * CELL_SIZE + CELL_SIZE // 2
                    self.pacman.current_y = self.pacman.target_y * CELL_SIZE + CELL_SIZE // 2
                    self.pacman.x = self.pacman.target_x
                    self.pacman.y = self.pacman.target_y
            else:
                moved = False  # 仍在移動中，未到達目標

        try:
            self.update(FPS, move_pacman)  # 更新遊戲狀態
        except Exception as e:
            print(f"遊戲更新失敗：{str(e)}")
            raise RuntimeError(f"遊戲更新失敗：{str(e)}")
        
        self.old_score = self.current_score  # 保存當前分數
        if moved:
            self.current_score = self.pacman.score  # 更新分數
        reward = self.current_score - self.old_score  # 計算獎勵（分數增量）
        # if wall_collision:
        #     reward -= 1  # 撞牆懲罰
        # min_ghost_dist = min(((self.pacman.current_x - g.current_x) ** 2 + 
        #                 (self.pacman.current_y - g.current_y) ** 2) ** 0.5 
        #                 for g in self.ghosts)
        # if min_ghost_dist < CELL_SIZE * 2:
        #     reward -= 10
        if not self.power_pellets and not self.score_pellets:
            reward += 500
        truncated = False
        if self.game_over:
            truncated = True  # 遊戲結束時截斷
        terminated = self.game_over  # 遊戲結束時終止
        next_state = np.array(self._get_state(), dtype=np.float32)  # 獲取下一狀態
        info = {
            "frame_count": self.frame_count,
            "current_score": self.current_score,
            "eaten_pellets": self.eaten_pellets,
            "total_pellets": self.total_pellets,
            "valid_step": moved and not terminated  # 有效步驟條件
        }
        
        self.frame_count += 1  # 增加幀數
        
        return next_state, reward, terminated, truncated, info

    def close(self):
        """
        關閉環境，清理資源。

        原理：
        - 調用父類的 end_game 方法，釋放遊戲相關資源。
        - 確保環境在訓練結束後正確關閉，防止資源洩漏。
        """
        super().end_game()
        print("環境已關閉")