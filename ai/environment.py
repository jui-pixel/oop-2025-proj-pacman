import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from gym.spaces import Discrete, Box
from game.game import Game
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, CELL_SIZE, FPS

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
        self.width = width
        self.height = height
        self.cell_size = CELL_SIZE
        self.seed = seed
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
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
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
        self.frame_count = 0
        state = self._get_state()
        print(f"Reset: Pac-Man at ({self.pacman.x}, {self.pacman.y}), Ghosts at {[f'({g.x}, {g.y})' for g in self.ghosts]}, Lives={self.pacman.lives}")
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        """
        執行一步動作，更新遊戲狀態並返回結果。

        Args:
            action (int): 動作（0=上, 1=下, 2=左, 3=右）。

        Returns:
            tuple: (下一狀態, 獎勵, 是否終止, 是否截斷, 資訊字典)。
        """
        if not 0 <= action < 4:
            print(f"Invalid action: {action}")
            raise ValueError(f"Invalid action: {action}")

        old_score = self.current_score
        old_pellets_count = len(self.power_pellets) + len(self.score_pellets)
        old_x, old_y = self.pacman.x, self.pacman.y

        def move_pacman():
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = directions[action]
            self.pacman.set_new_target(dx, dy, self.maze)

        try:
            self.update(FPS, move_pacman)  # 直接調用 Game 的 update 方法
        except Exception as e:
            print(f"Game update failed: {str(e)}")
            raise RuntimeError(f"Game update failed: {str(e)}")

        self.current_score = self.pacman.score
        new_pellets_count = len(self.power_pellets) + len(self.score_pellets)

        # 計算獎勵：基於分數變化，減去時間懲罰
        reward = self.current_score - old_score
        reward -= 0.01  # 時間懲罰

        # 檢查遊戲結束條件
        if not self.power_pellets and not self.score_pellets:
            self.game_over = True
            print("All pellets eaten, game won")
        elif self.pacman.lives <= 0:
            self.game_over = True
            print(f"Game over, no lives left, final score: {self.current_score}")

        # 記錄與最近球的距離
        min_pellet_dist = min(
            [((self.pacman.x - pellet.x) ** 2 + (self.pacman.y - pellet.y) ** 2) ** 0.5 
             for pellet in self.score_pellets + self.power_pellets] + [float('inf')]
        )
        print(f"Distance to nearest pellet: {min_pellet_dist:.2f}")

        # 碰撞檢測（用於終端輸出）
        collision_detected = False
        for ghost in self.ghosts:
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < self.cell_size / 2:
                if ghost.edible and ghost.edible_timer > 0:
                    print(f"Ate edible ghost at ({ghost.x}, {ghost.y}), Score increased by {self.current_score - old_score}")
                elif not ghost.edible and not ghost.returning_to_spawn:
                    collision_detected = True
                    print(f"Collision with ghost at ({ghost.x}, {ghost.y}), Lives left: {self.pacman.lives}")

        next_state = np.array(self._get_state(), dtype=np.float32)
        terminated = self.game_over
        truncated = False
        info = {
            'score': self.current_score,
            'game_over': self.game_over,
            'eaten_pellets': self.total_pellets - new_pellets_count,
            'total_pellets': self.total_pellets,
            'lives': self.pacman.lives,
            'collision': collision_detected,
            'min_pellet_dist': min_pellet_dist
        }
        self.frame_count += 1
        print(f"Step: Action={action}, Reward={reward:.2f}, Lives={self.pacman.lives}, Terminated={terminated}, Score={self.current_score}")
        return next_state, reward, terminated, truncated, info

    def close(self):
        """
        關閉環境，清理資源。
        """
        super().end_game()
        print("Environment closed")