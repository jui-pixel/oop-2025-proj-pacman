import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from gym.spaces import Discrete, Box
from game.game import Game
from config import *
from typing import Callable

class PacManEnv(Game):
    metadata = {"render_modes": [], "render_fps": None}

    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        """
        初始化 Pac-Man 環境，繼承 Game 類。

        Args:
            width (int): 迷宮寬度。
            height (int): 迷宮高度。
            seed (int): 隨機種子。
        """
        super().__init__(player_name="RL_Agent")
        self.pacman.speed = 5 * self.pacman.speed
        self.width = width
        self.height = height
        self.cell_size = CELL_SIZE
        self.seed = seed
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
        self.old_score = 0
        self.frame_count = 0
        self.state_channels = 6
        self.state_shape = (self.state_channels, self.height, self.width)

        self.action_space = Discrete(4)  # 上、下、左、右
        self.observation_space = Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)

        np.random.seed(seed)
        print(f"Initialized PacManEnv: width={width}, height={height}, seed={seed}, lives={self.pacman.lives}")

    def _get_state(self):
        """
        獲取當前遊戲狀態，6 個通道表示 Pac-Man、能量球、分數球、可食用鬼魂、普通鬼魂和牆壁。

        Returns:
            np.ndarray: 狀態張量，形狀為 (6, height, width)。
        """
        state = np.zeros((self.state_channels, self.height, self.width), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.get_tile(x, y) in ['#', 'X']:
                    state[5, y, x] = 1.0
        state[0, self.pacman.y, self.pacman.x] = 1.0
        for pellet in self.power_pellets:
            state[1, pellet.y, pellet.x] = 1.0
        for pellet in self.score_pellets:
            state[2, pellet.y, pellet.x] = 1.0
        for ghost in self.ghosts:
            if ghost.edible and ghost.edible_timer > 0 and not ghost.returning_to_spawn:
                state[3, ghost.y, ghost.x] = 1.0
            else:
                state[4, ghost.y, ghost.x] = 1.0
        # print(f"State shape from _get_state: {state.shape}")
        return state

    def reset(self, seed=None):
        """
        重置環境，重新初始化遊戲狀態。

        Args:
            seed (int, optional): 隨機種子。

        Returns:
            tuple: (狀態張量, 資訊字典)。
        """
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        super().__init__(player_name="RL_Agent")  # 調用 Game 的 __init__ 重置遊戲
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
        更新遊戲狀態，包括移動 Pac-Man、鬼魂和檢查碰撞。

        Args:
            fps (int): 每秒幀數，用於計時。
            move_pacman (Callable[[], None]): 控制 Pac-Man 移動的函數。
        """

        move_pacman()  # 移動 Pac-Man

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

        if not self.power_pellets and not self.score_pellets:
            print(f"Game Won! All pellets collected. Final Score: {self.pacman.score}")
            self.running = False  # 遊戲結束
            self.game_over = True
            

    def _check_collision(self, fps: int) -> None:
        """
        檢查 Pac-Man 與鬼魂的碰撞，根據鬼魂狀態更新分數或觸發死亡動畫。

        Args:
            fps (int): 每秒幀數，用於計時。
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
                    for g in self.ghosts:  # 所有鬼魂返回重生點
                        g.set_returning_to_spawn(fps)
                    if self.pacman.lives <= 0:
                        self.running = False  # 遊戲結束
                        self.game_over = True 
                        # print(f"Game Over! Score: {self.pacman.score}")
                    else:
                        # print(f"Life lost! Remaining lives: {self.pacman.lives}")
                        break
                    break
                
    def step(self, action):
        """
        執行一步動作，更新遊戲狀態並返回結果。

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
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = directions[action]
            new_x, new_y = self.pacman.x + dx, self.pacman.y + dy
            if self.maze.get_tile(new_x, new_y) not in ['#', 'X']:
                if self.pacman.move_towards_target(FPS):  # 僅在到達當前目標時更新
                    self.pacman.set_new_target(dx, dy, self.maze)
                elif not self.pacman.move_towards_target(FPS):  # 繼續朝當前目標移動
                    moved = False
            else:
                wall_collision = True  # 檢測到撞牆
        try:
            self.update(FPS, move_pacman)
        except Exception as e:
            print(f"Game update failed: {str(e)}")
            raise RuntimeError(f"Game update failed: {str(e)}")
        self.old_score = self.current_score
        if moved:
            self.current_score = self.pacman.score  
        reward = self.current_score - self.old_score
        if wall_collision:
            reward -= 5
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