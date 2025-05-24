# game/environment.py
"""
定義 Pac-Man 遊戲的強化學習環境，遵循 OpenAI Gym 規範。
負責初始化遊戲迷宮、管理狀態轉換、計算獎勵並提供狀態觀察。
"""

import numpy as np
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

class PacManEnv:
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        """
        初始化 Pac-Man 遊戲環境，設置迷宮和遊戲實體。

        Args:
            width (int): 迷宮寬度，預設從 config 導入。
            height (int): 迷宮高度，預設從 config 導入。
            seed (int): 隨機種子，用於生成一致的迷宮，預設從 config 導入。
        """
        self.maze = Map(w=width, h=height, seed=seed)  # 初始化迷宮
        self.maze.generate_maze()  # 生成迷宮結構
        # 初始化遊戲實體：Pac-Man、鬼魂、能量球、分數球
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.done = False  # 遊戲結束標誌，初始為 False
        self.action_space = [0, 1, 2, 3]  # 動作空間：上(0)、下(1)、左(2)、右(3)
        self.observation_space = (height, width, 6)  # 狀態空間：高度、寬度、6 通道

    def reset(self):
        """
        重置環境，重新生成迷宮和實體，恢復初始狀態。

        Returns:
            numpy.ndarray: 初始狀態，形狀為 (height, width, 6)。
        """
        self.maze.generate_maze()  # 重新生成迷宮
        # 重新初始化所有遊戲實體
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.done = False  # 重置結束標誌
        return self._get_state()  # 返回初始狀態

    def _get_state(self):
        """
        構建當前遊戲狀態，包含 Pac-Man、能量球、分數球、鬼魂和牆壁的二維表示。

        Returns:
            numpy.ndarray: 當前狀態，形狀為 (height, width, 6)，6 通道分別表示：
                0: Pac-Man 位置
                1: 能量球位置
                2: 分數球位置
                3: 可吃鬼魂位置
                4: 不可吃鬼魂位置
                5: 牆壁位置
        """
        state = np.zeros((self.maze.h, self.maze.w, 6), dtype=np.float32)
        state[self.pacman.x, self.pacman.y, 0] = 1.0  # 標記 Pac-Man 位置
        for pellet in self.power_pellets:
            state[pellet.x, pellet.y, 1] = 1.0  # 標記能量球
        for pellet in self.score_pellets:
            state[pellet.x, pellet.y, 2] = 1.0  # 標記分數球
        for ghost in self.ghosts:
            if ghost.edible and ghost.respawn_timer > 0:
                state[ghost.x, ghost.y, 3] = 1.0  # 標記可吃鬼魂
            else:
                state[ghost.x, ghost.y, 4] = 1.0  # 標記不可吃鬼魂
        # 標記牆壁位置
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
                - 新狀態 (numpy.ndarray): 形狀為 (height, width, 6)。
                - 獎勵 (float): 當前步的獎勵值。
                - 是否結束 (bool): 遊戲是否結束。
                - 附加資訊 (dict): 目前為空，保留擴展用。
        """
        # 將動作映射為方向向量
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        moving = self.pacman.set_new_target(dx, dy, self.maze)  # 設置 Pac-Man 新目標
        if moving:
            self.pacman.move_towards_target(self.maze)  # 移動 Pac-Man

        # 計算獎勵
        reward = -0.1  # 基本步進懲罰，鼓勵快速完成
        if self.pacman.eat_pellet(self.power_pellets) > 0:
            reward = 40  # 吃能量球獎勵
        if self.pacman.eat_score_pellet(self.score_pellets) > 0:
            reward = 10  # 吃分數球獎勵
        
        min_power_dist = float('inf') if not self.power_pellets else min(
            abs(self.pacman.x - p.x) + abs(self.pacman.y - p.y) for p in self.power_pellets)
        min_score_dist = float('inf') if not self.score_pellets else min(
            abs(self.pacman.x - p.x) + abs(self.pacman.y - p.y) for p in self.score_pellets)
        reward += max(0, 1.0 - min_power_dist * 0.1)  # 接近能量球獎勵
        reward += max(0, 0.5 - min_score_dist * 0.1)  # 接近分數球獎勵

        # 計算與鬼魂的距離，遠離不可吃鬼魂獲得小獎勵
        min_ghost_dist = float('inf')
        for ghost in self.ghosts:
            if not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                dist = abs(self.pacman.x - ghost.x) + abs(self.pacman.y - ghost.y)
                min_ghost_dist = min(min_ghost_dist, dist)
        if min_ghost_dist > 5:
            reward += 0.5  # 遠離鬼魂獎勵

        # 檢查與鬼魂的碰撞
        for ghost in self.ghosts:
            if self.pacman.x == ghost.x and self.pacman.y == ghost.y:
                if ghost.edible and ghost.respawn_timer > 0:
                    # 吃可吃鬼魂，根據死亡次數給不同獎勵
                    reward = [100, 150, 250, 400][min(ghost.death_count, 3)]
                    ghost.set_returning_to_spawn(30)  # 鬼魂進入重生模式
                elif not ghost.returning_to_spawn and not ghost.waiting:
                    reward = -50  # 被不可吃鬼魂抓住，遊戲結束
                    self.done = True
        
        # 檢查是否吃完所有分數球和能量球，給予 1000 分獎勵
        if len(self.power_pellets) == 0 and len(self.score_pellets) == 0:
            reward += 1000  # 吃完所有物品的額外獎勵
            self.done = True  # 任務完成，結束遊戲
        
        # 更新鬼魂狀態
        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == 'S':
                    ghost.set_waiting(30)  # 鬼魂到達重生點，進入等待模式
                else:
                    ghost.move(self.pacman, self.maze, 30)  # 鬼魂移動邏輯

        return self._get_state(), reward, self.done, {}  # 返回結果

    def render(self):
        """
        渲染遊戲環境（由 main.py 實現，僅保留接口）。
        """
        pass