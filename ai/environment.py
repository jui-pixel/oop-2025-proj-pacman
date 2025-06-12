# ai/environment.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 將父目錄加入路徑
import numpy as np
from gym.spaces import Discrete, Box
from game.game import Game
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, CELL_SIZE, FPS, EDIBLE_DURATION, GHOST_SCORES, TILE_PATH, TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN
import random
from typing import Callable

class PacManEnv(Game):
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED, ghost_penalty_weight=3.0):
        """
        初始化 Pac-Man 環境，提供強化學習接口。

        Args:
            width (int): 迷宮寬度。
            height (int): 迷宮高度。
            seed (int): 隨機種子。
            ghost_penalty_weight (float): 鬼魂距離懲罰權重。
        """
        super().__init__(player_name="RL_Agent")  # 固定 4 隻鬼魂（由 Game 類控制）
        self.width = width
        self.height = height
        self.cell_size = CELL_SIZE
        self.seed = seed
        self.ghost_penalty_weight = ghost_penalty_weight
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
        self.old_score = 0
        self.frame_count = 0
        self.ghost_move_counter = 2
        self.state_channels = 6  # 6 通道狀態
        self.state_shape = (self.state_channels, self.height, self.width)
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)
        np.random.seed(seed)
        print(f"初始化 PacManEnv：寬度={width}，高度={height}，種子={seed}，鬼魂數=4")

    def _get_state(self):
        """
        獲取遊戲狀態，返回 6 通道張量：
        - 通道 0：Pac-Man 位置
        - 通道 1：能量球
        - 通道 2：分數球
        - 通道 3：可食用鬼魂
        - 通道 4：普通鬼魂
        - 通道 5：牆壁
        """
        state = np.zeros(self.state_shape, dtype=np.float32)
        # 設置牆壁
        for y in range(self.height):
            for x in range(self.width):
                if self.maze.get_tile(x, y) in [TILE_BOUNDARY,TILE_WALL]:
                    state[5, y, x] = 1.0
        # 設置 Pac-Man
        state[0, self.pacman.y, self.pacman.x] = 1.0
        # 設置能量球和分數球
        for pellet in self.power_pellets:
            state[1, pellet.y, pellet.x] = 1.0
        for pellet in self.score_pellets:
            state[2, pellet.y, pellet.x] = 1.0
        # 設置鬼魂
        for ghost in self.ghosts:
            if ghost.returning_to_spawn or ghost.waiting:
                    continue
            elif ghost.edible:
                state[3, ghost.y, ghost.x] = 1.0
            else:
                state[4, ghost.y, ghost.x] = 1.0
        return state

    def get_expert_action(self):
        """
        使用規則基礎 AI 提供專家動作。
        """
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        success = self.pacman.rule_based_ai_move(self.maze, self.power_pellets, self.score_pellets, self.ghosts)
        if success and self.pacman.last_direction:
            dx, dy = self.pacman.last_direction
            for i, (dx_dir, dy_dir) in enumerate(directions):
                if dx == dx_dir and dy == dy_dir:
                    return i
        safe_directions = [i for i, (dx, dy) in enumerate(directions) 
                          if self.maze.xy_valid(self.pacman.x + dx, self.pacman.y + dy)
                          and self.maze.get_tile(self.pacman.x + dx, self.pacman.y + dy) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
        return random.choice(safe_directions) if safe_directions else 0

    def reset(self, seed=MAZE_SEED, random_spawn_seed=0):
        """
        重置環境，重新初始化遊戲。
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        super().__init__(player_name="RL_Agent")  # 固定 4 隻鬼魂
        if random_spawn_seed != 0:
            random.seed(self.seed + random_spawn_seed)
            valid_positions = [(x, y) for y in range(1, self.maze.height - 1) for x in range(1, self.maze.width - 1)
                              if self.maze.get_tile(x, y) == TILE_PATH]
            self.pacman.x, self.pacman.y = random.choice(valid_positions)
            self.pacman.initial_x = self.pacman.x
            self.pacman.initial_y = self.pacman.y
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
        self.old_score = 0
        self.frame_count = 0
        self.ghost_move_counter = 2
        state = self._get_state()
        return np.array(state, dtype=np.float32), {}

    def update(self, fps: int, move_pacman: Callable[[], None]) -> None:
        """
        更新遊戲狀態。
        """
        move_pacman()
        score_from_pellet = self.pacman.eat_pellet(self.power_pellets)
        if score_from_pellet > 0:
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)
        self.pacman.eat_score_pellet(self.score_pellets)
        for ghost in self.ghosts:
            if ghost.move_towards_target(FPS):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == TILE_GHOST_SPAWN:
                    ghost.set_waiting(FPS)
                elif ghost.returning_to_spawn:
                    ghost.return_to_spawn(self.maze)
                else:
                    if self.frame_count % self.ghost_move_counter == 0:
                        ghost.move(self.pacman, self.maze, FPS)
                        ghost.current_x = ghost.target_x * CELL_SIZE + CELL_SIZE // 2
                        ghost.current_y = ghost.target_y * CELL_SIZE + CELL_SIZE // 2
                        ghost.x = ghost.target_x
                        ghost.y = ghost.target_y
        self._check_collision(FPS)
        if not self.power_pellets and not self.score_pellets:
            print(f"遊戲勝利！最終分數：{self.pacman.score}")
            self.running = False
            self.game_over = True

    def _check_collision(self, fps: int) -> None:
        """
        檢查 Pac-Man 與鬼魂的碰撞。
        """
        if not self.running:
            return
        for ghost in self.ghosts:
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    ghost.set_returning_to_spawn(FPS)
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.pacman.lose_life(self.maze)
                    for g in self.ghosts:
                        g.set_returning_to_spawn(FPS)
                    if self.pacman.lives <= 0:
                        self.running = False
                        self.game_over = True
                    else:
                        break

    def step(self, action):
        """
        執行一步動作，返回五元組。
        """
        if not 0 <= action < 4:
            raise ValueError(f"無效動作：{action}")
        moved = True
        wall_collision = False
        def move_pacman():
            nonlocal moved, wall_collision
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = directions[action]
            if self.pacman.move_towards_target(FPS):
                if not self.pacman.set_new_target(dx, dy, self.maze):
                    wall_collision = True
                else:
                    self.pacman.current_x = self.pacman.target_x * CELL_SIZE + CELL_SIZE // 2
                    self.pacman.current_y = self.pacman.target_y * CELL_SIZE + CELL_SIZE // 2
                    self.pacman.x = self.pacman.target_x
                    self.pacman.y = self.pacman.target_y
            else:
                moved = False
        try:
            self.update(FPS, move_pacman)
        except Exception as e:
            raise RuntimeError(f"遊戲更新失敗：{str(e)}")
        if moved:
            self.current_score = self.pacman.score
        reward = (self.current_score - self.old_score) * 5
        self.old_score = self.current_score
        if wall_collision:
            reward -= 5
        ghost_penalty = 0
        for ghost in self.ghosts:
            dist = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if ghost.returning_to_spawn or ghost.waiting:
                continue
            elif ghost.edible:
                if dist < 8 * CELL_SIZE:
                    ghost_penalty -= (self.ghost_penalty_weight / (dist / CELL_SIZE + 0.1) / 2)
            else:
                if dist < 5 * CELL_SIZE:
                    ghost_penalty += self.ghost_penalty_weight / (dist / CELL_SIZE + 0.1)
        reward -= ghost_penalty
        if not self.game_over:
            reward += 0.1  # 存活獎勵
        if not self.power_pellets and not self.score_pellets:
            reward += 5000
        truncated = self.game_over
        terminated = self.game_over
        next_state = np.array(self._get_state(), dtype=np.float32)
        info = {
            "frame_count": self.frame_count,
            "current_score": self.current_score,
            "eaten_pellets": self.eaten_pellets,
            "total_pellets": self.total_pellets,
            "valid_step": moved and not terminated,
            "lives_lost": self.pacman.lives < 3
        }
        self.frame_count += 1
        return next_state, reward, terminated, info

    def close(self):
        """
        關閉環境。
        """
        super().end_game()
        print("環境已關閉")