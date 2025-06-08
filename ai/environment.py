# ai/environment.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 加入父目錄到路徑
import numpy as np
from gym.spaces import Discrete, Box
from game.game import Game
from config import *
from typing import Callable

class PacManEnv(Game):
    metadata = {"render_modes": [], "render_fps": None}  # 元數據，暫無渲染功能

    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        """
        初始化 Pac-Man 環境，基於 Game 類。

        Args:
            width (int): 迷宮寬度，預設從 config 載入。
            height (int): 迷宮高度，預設從 config 載入。
            seed (int): 隨機種子，確保可重現性。
        """
        super().__init__(player_name="RL_Agent")  # 調用父類初始化
        self.pacman.speed = 5 * self.pacman.speed
        self.width = width
        self.height = height
        self.cell_size = CELL_SIZE
        self.seed = seed
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)  # 總彈丸數
        self.eaten_pellets = 0  # 已吃彈丸數
        self.game_over = False  # 遊戲是否結束
        self.current_score = 0  # 當前分數
        self.old_score = 0  # 上一次分數
        self.frame_count = 0  # 幀數計數
        self.state_channels = 6  # 狀態通道數（Pac-Man、能量球等）
        self.state_shape = (self.state_channels, self.height, self.width)  # 狀態形狀

        self.action_space = Discrete(4)  # 動作空間：0=上, 1=下, 2=左, 3=右
        self.observation_space = Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)  # 狀態空間

        np.random.seed(seed)  # 設置隨機種子
        print(f"Initialized PacManEnv: width={width}, height={height}, seed={seed}, lives={self.pacman.lives}")

    def _get_state(self):
        """
        獲取當前遊戲狀態，生成 6 通道的張量表示。

        Returns:
            np.ndarray: 狀態張量，形狀為 (6, height, width)，分別表示：
                0: Pac-Man 位置，1: 能量球，2: 分數球，3: 可食用鬼魂，4: 普通鬼魂，5: 牆壁。
        """
        state = np.zeros((self.state_channels, self.height, self.width), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.get_tile(x, y) in ['#', 'X']:  # 牆壁和邊界
                    state[5, y, x] = 1.0
        state[0, self.pacman.y, self.pacman.x] = 1.0  # Pac-Man 位置
        for pellet in self.power_pellets:
            state[1, pellet.y, pellet.x] = 1.0  # 能量球
        for pellet in self.score_pellets:
            state[2, pellet.y, pellet.x] = 1.0  # 分數球
        for ghost in self.ghosts:
            if ghost.edible and ghost.edible_timer > 0 and not ghost.returning_to_spawn:
                state[3, ghost.y, ghost.x] = 1.0  # 可食用鬼魂
            else:
                state[4, ghost.y, ghost.x] = 1.0  # 普通鬼魂
        return state

    def reset(self, seed=None):
        """
        重置環境，重新開始遊戲。

        Args:
            seed (int, optional): 隨機種子，若提供則更新。

        Returns:
            tuple: (狀態張量, 資訊字典)。
        """
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        super().__init__(player_name="RL_Agent")  # 重置遊戲
        self.pacman.speed = 5 * self.pacman.speed
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
        self.old_score = 0
        self.frame_count = 0
        state = self._get_state()
        return np.array(state, dtype=np.float32), {}

    def update(self, fps: int, move_pacman: Callable[[], None]) -> None:
        """
        更新遊戲狀態，包括移動 Pac-Man 和鬼魂。

        Args:
            fps (int): 每秒幀數，用於時間計算。
            move_pacman (Callable[[], None]): 控制 Pac-Man 移動的函數。
        """
        move_pacman()  # 執行 Pac-Man 移動

        # 檢查是否吃到能量球
        score_from_pellet = self.pacman.eat_pellet(self.power_pellets)
        if score_from_pellet > 0:
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)  # 設置鬼魂為可吃狀態

        # 檢查是否吃到分數球
        self.pacman.eat_score_pellet(self.score_pellets)

        # 移動所有鬼魂
        for ghost in self.ghosts:
            if ghost.move_towards_target(FPS):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == TILE_GHOST_SPAWN:
                    ghost.set_waiting(fps)  # 鬼魂到達重生點後等待
                elif ghost.returning_to_spawn:
                    ghost.return_to_spawn(self.maze)
                else:
                    ghost.move(self.pacman, self.maze, fps)  # 執行鬼魂移動邏輯

        # 檢查碰撞
        self._check_collision(fps)

        if not self.power_pellets and not self.score_pellets:  # 所有彈丸吃完
            print(f"Game Won! All pellets collected. Final Score: {self.pacman.score}")
            self.running = False  # 遊戲結束
            self.game_over = True

    def _check_collision(self, fps: int) -> None:
        """
        檢查 Pac-Man 與鬼魂的碰撞，更新分數或觸發死亡。

        Args:
            fps (int): 每秒幀數。
        """
        if not self.running:
            return
        
        for ghost in self.ghosts:
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:  # 碰撞檢測
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]  # 增加分數
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    ghost.set_returning_to_spawn(fps)  # 鬼魂返回重生點
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.pacman.lose_life(self.maze)
                    for g in self.ghosts:  # 所有鬼魂返回
                        g.set_returning_to_spawn(fps)
                    if self.pacman.lives <= 0:
                        self.running = False  # 遊戲結束
                        self.game_over = True
                    else:
                        break

    def step(self, action):
        """
        執行一步動作，更新狀態並返回結果。

        Args:
            action (int): 動作（0=上, 1=下, 2=左, 3=右）。

        Returns:
            tuple: (下一狀態, 獎勵, 是否終止, 是否截斷, 資訊字典)。
        """
        if not 0 <= action < 4:
            raise ValueError(f"Invalid action: {action}")

        moved = True
        wall_collision = False
        
        def move_pacman():
            nonlocal moved, wall_collision
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上、下、左、右
            dx, dy = directions[action]
            if self.pacman.move_towards_target(FPS):  # 僅在到達目標時更新
                if not self.pacman.set_new_target(dx, dy, self.maze):
                    wall_collision = True
                self.pacman.x = self.pacman.target_x
                self.pacman.y = self.pacman.target_y
            else:  # 繼續朝目標移動
                moved = False

        try:
            self.update(FPS, move_pacman)
        except Exception as e:
            print(f"Game update failed: {str(e)}")
            raise RuntimeError(f"Game update failed: {str(e)}")
        self.old_score = self.current_score
        if moved:
            self.current_score = self.pacman.score  
        reward = self.current_score - self.old_score  # 獎勵為分數增量
        if wall_collision:
            reward -= 5  # 撞牆懲罰
        truncated = False
        if self.game_over:
            truncated = True

        next_state = np.array(self._get_state(), dtype=np.float32)
        info = {
            "frame_count": self.frame_count,
            "current_score": self.current_score,
            "eaten_pellets": self.eaten_pellets,
            "total_pellets": self.total_pellets
        }
        terminated = self.game_over
        self.frame_count += 1
        
        if not moved and not terminated:
            reward = 0  # 移動中無獎勵
            info["valid_step"] = False
        else:
            info["valid_step"] = True
        
        return next_state, reward, terminated, truncated, info

    def close(self):
        """
        關閉環境，清理資源。
        """
        super().end_game()
        print("Environment closed")