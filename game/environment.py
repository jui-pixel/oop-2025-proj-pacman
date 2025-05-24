# game/environment.py
"""
定義 Pac-Man 遊戲的強化學習環境，符合 OpenAI Gym 規範。
負責初始化遊戲環境、管理狀態轉換和獎勵計算。
"""
import numpy as np
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

class PacManEnv:
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        """
        初始化環境，設置迷宮和遊戲實體。
        
        Args:
            width (int): 迷宮寬度。
            height (int): 迷宮高度。
            seed (int): 隨機種子，用於生成一致的迷宮。
        """
        self.maze = Map(w=width, h=height, seed=seed)
        self.maze.generate_maze()
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.done = False  # 遊戲結束標誌
        self.action_space = [0, 1, 2, 3]  # 動作空間：上(0)、下(1)、左(2)、右(3)
        self.observation_space = (height, width, 6)  # 觀察空間：(高度, 寬度, 6個通道)

    def reset(self):
        """
        重置環境，重新生成迷宮和實體。
        
        Returns:
            初始狀態（numpy 陣列）。
        """
        self.maze.generate_maze()
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        獲取當前遊戲狀態，包含 Pac-Man、能量球、分數球、鬼魂和牆壁位置。
        
        Returns:
            numpy 陣列，形狀為 (height, width, 6)，表示遊戲狀態。
        """
        state = np.zeros((self.maze.h, self.maze.w, 6), dtype=np.float32)
        state[self.pacman.x, self.pacman.y, 0] = 1.0  # Pac-Man 位置
        for pellet in self.power_pellets:
            state[pellet.x, pellet.y, 1] = 1.0  # 能量球
        for pellet in self.score_pellets:
            state[pellet.x, pellet.y, 2] = 1.0  # 分數球
        for ghost in self.ghosts:
            if ghost.edible and ghost.respawn_timer > 0:
                state[ghost.x, ghost.y, 3] = 1.0  # 可吃鬼魂
            else:
                state[ghost.x, ghost.y, 4] = 1.0  # 不可吃鬼魂
        # 加入牆壁位置 (通道 5)
        for y in range(self.maze.h):
            for x in range(self.maze.w):
                if self.maze.get_tile(x, y) in ['#', 'X', 'D']:  # 表示牆壁
                    state[x, y, 5] = 1.0
        return state

    def step(self, action):
        """
        執行一步動作，更新環境並返回新狀態、獎勵和結束標誌。
        
        Args:
            action (int): 動作索引（0: 上, 1: 下, 2: 左, 3: 右）。
        
        Returns:
            Tuple: (新狀態, 獎勵, 是否結束, 附加資訊)
        """
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        moving = self.pacman.set_new_target(dx, dy, self.maze)
        if moving:
            self.pacman.move_towards_target(self.maze)
        
        # 計算獎勵
        reward = 0.5  # 基本步進獎勵
        if self.pacman.eat_pellet(self.power_pellets) > 0:
            reward = 40  # 吃能量球獎勵
        if self.pacman.eat_score_pellet(self.score_pellets) > 0:
            reward = 10   # 吃分數球獎勵
            
        # 計算與最近能量球和分數球的距離獎勵
        min_power_dist = float('inf') if not self.power_pellets else min(
            abs(self.pacman.x - p.x) + abs(self.pacman.y - p.y) for p in self.power_pellets)
        min_score_dist = float('inf') if not self.score_pellets else min(
            abs(self.pacman.x - p.x) + abs(self.pacman.y - p.y) for p in self.score_pellets)
        reward += max(0, 1.0 - min_power_dist * 0.1)  # 接近能量球給予獎勵
        reward += max(0, 0.5 - min_score_dist * 0.1)  # 接近分數球給予獎勵
        
        # 檢查與鬼魂的碰撞
        for ghost in self.ghosts:
            if self.pacman.x == ghost.x and self.pacman.y == ghost.y:
                if ghost.edible and ghost.respawn_timer > 0:
                    reward = [100, 150, 250, 400][min(ghost.death_count, 3)]  # 吃鬼魂獎勵
                    ghost.set_returning_to_spawn(30)
                elif not ghost.returning_to_spawn and not ghost.waiting:
                    reward = -100  # 被鬼魂抓住，遊戲結束
                    self.done = True
        
        # 更新鬼魂
        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == 'S':
                    ghost.set_waiting(30)
                else:
                    ghost.move(self.pacman, self.maze, 30)
        
        return self._get_state(), reward, self.done, {}
    
    def render(self):
        """渲染環境（由 main.py 處理）。"""
        pass