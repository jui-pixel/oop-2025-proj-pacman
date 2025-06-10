# game/game.py
"""
定義 Pac-Man 遊戲的核心邏輯，包括初始化、更新狀態和碰撞檢測。
"""

from typing import List, Tuple, Optional, Callable
from .entities.pacman import PacMan
from .entities.ghost import *
from .entities.entity_initializer import initialize_entities
from .entities.pellets import PowerPellet, ScorePellet
from .maze_generator import Map
from config import EDIBLE_DURATION, GHOST_SCORES, MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, FPS, CELL_SIZE, TILE_GHOST_SPAWN
import config
from collections import deque
import pygame

class Game:
    def __init__(self, player_name: str):
        """
        初始化遊戲，設置迷宮、Pac-Man、鬼魂和其他實體。

        原理：
        - 創建遊戲實例，初始化迷宮、Pac-Man、鬼魂、能量球和分數球。
        - 使用指定的迷宮寬高和種子生成隨機迷宮，確保每次遊戲地圖一致。
        - 記錄遊戲開始時間，用於計算遊玩時長。
        - 設置死亡動畫相關屬性，控制遊戲結束時的視覺效果。

        Args:
            player_name (str): 玩家名稱，用於記錄分數。
        """
        self.seed = config.MAZE_SEED
        self.maze = Map(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=self.seed)  # 初始化迷宮
        self.maze.generate_maze()  # 生成隨機迷宮
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = self._initialize_entities()  # 初始化所有實體
        self.respawn_points = [(x, y) for y in range(self.maze.height) for x in range(self.maze.width) 
                               if self.maze.get_tile(x, y) == TILE_GHOST_SPAWN]  # 收集鬼魂重生點坐標
        self.ghost_score_index = 0  # 鬼魂分數索引，追蹤連續吃鬼魂的分數遞增
        self.running = True  # 遊戲運行狀態
        self.player_name = player_name  # 玩家名稱
        self.start_time = pygame.time.get_ticks()  # 記錄遊戲開始時間（毫秒）
        self.death_animation = False  # 死亡動畫狀態
        self.death_animation_timer = 0  # 死亡動畫計時器（幀數）
        self.death_animation_duration = FPS  # 死亡動畫持續時間（預設 60 幀，相當於 1 秒）

    def _initialize_entities(self) -> Tuple[PacMan, List[Ghost], List[PowerPellet], List[ScorePellet]]:
        """
        初始化遊戲中的所有實體（Pac-Man、鬼魂、能量球、分數球）。

        原理：
        - 調用 entity_initializer 模塊的 initialize_entities 函數，根據迷宮結構生成實體。
        - 確保 Pac-Man、鬼魂和彈丸的初始位置合理且不重疊。
        - 返回一個包含所有實體的元組，供遊戲主循環使用。

        Returns:
            Tuple: (pacman, ghosts, power_pellets, score_pellets)
            - pacman: PacMan 物件。
            - ghosts: 鬼魂列表。
            - power_pellets: 能量球列表。
            - score_pellets: 分數球列表。
        """
        return initialize_entities(self.maze)

    def update(self, fps: int, move_pacman: Callable[[], None]) -> None:
        """
        更新遊戲狀態，包括移動 Pac-Man、鬼魂和檢查碰撞。

        原理：
        - 每幀調用一次，處理遊戲邏輯的更新，包括移動、吃彈丸和碰撞檢測。
        - 若死亡動畫正在播放，僅更新動畫計時器，不執行其他邏輯。
        - 當 Pac-Man 吃到能量球時，設置所有鬼魂為可食用狀態。
        - 當所有彈丸被吃完時，遊戲勝利並結束。
        - 鬼魂移動邏輯根據其狀態（追逐、逃跑、返回重生點）執行。

        Args:
            fps (int): 每秒幀數，用於計算每幀時間。
            move_pacman (Callable[[], None]): 控制 Pac-Man 移動的函數（玩家輸入或 AI）。
        """
        if self.death_animation:
            self.death_animation_timer += 1
            if self.death_animation_timer >= self.death_animation_duration:
                self.death_animation = False  # 動畫結束，重置狀態
            return

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
            if ghost.move_towards_target(FPS):  # 若鬼魂到達目標格子
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == TILE_GHOST_SPAWN:
                    ghost.set_waiting(fps)  # 到達重生點後進入等待狀態
                elif ghost.returning_to_spawn:
                    ghost.return_to_spawn(self.maze)  # 繼續返回重生點
                else:
                    ghost.move(self.pacman, self.maze, fps)  # 執行正常移動邏輯（追逐或逃跑）

        # 檢查碰撞
        self._check_collision(fps)

        # 檢查遊戲勝利條件
        if not self.power_pellets and not self.score_pellets:
            print(f"遊戲勝利！所有彈丸已收集。最終分數：{self.pacman.score}")
            self.running = False  # 遊戲結束

    def _check_collision(self, fps: int) -> None:
        """
        檢查 Pac-Man 與鬼魂的碰撞，根據鬼魂狀態更新分數或觸發死亡動畫。

        原理：
        - 使用歐幾里得距離檢測 Pac-Man 與鬼魂的碰撞，距離公式：dist = √((x1 - x2)^2 + (y1 - y2)^2)。
        - 若距離小於半個格子尺寸（CELL_SIZE / 2），則認為發生碰撞。
        - 碰撞後的行為取決於鬼魂狀態：
          - 可食用鬼魂：增加分數（根據 GHOST_SCORES 遞增），鬼魂返回重生點。
          - 不可食用鬼魂：Pac-Man 損失一條命，所有鬼魂返回重生點，若無生命則遊戲結束。

        Args:
            fps (int): 每秒幀數，用於設置鬼魂狀態。
        """
        if not self.running:
            return
        
        for ghost in self.ghosts:
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:  # 碰撞檢測
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]  # 增加分數
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)  # 更新分數索引
                    ghost.set_returning_to_spawn(fps)  # 鬼魂返回重生點
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.pacman.lose_life(self.maze)  # Pac-Man 損失一條命
                    for g in self.ghosts:  # 所有鬼魂返回重生點
                        g.set_returning_to_spawn(fps)
                    if self.pacman.lives <= 0:
                        self.running = False  # 遊戲結束
                        print(f"遊戲結束！分數：{self.pacman.score}")
                        self.death_animation = True  # 觸發死亡動畫
                        self.death_animation_timer = 0
                    else:
                        print(f"損失一條命！剩餘生命：{self.pacman.lives}")
                    break

    def is_running(self) -> bool:
        """
        檢查遊戲是否正在運行。

        原理：
        - 檢查遊戲運行狀態和 Pac-Man 的生命數，若 running 為 True 且生命數大於 0，則遊戲繼續。

        Returns:
            bool: 遊戲運行狀態。
        """
        return self.running and self.pacman.lives > 0

    def is_death_animation_playing(self) -> bool:
        """
        檢查死亡動畫是否正在播放。

        原理：
        - 返回 death_animation 屬性，指示當前是否正在播放 Pac-Man 的死亡動畫。

        Returns:
            bool: 死亡動畫狀態。
        """
        return self.death_animation

    def get_death_animation_progress(self) -> float:
        """
        獲取死亡動畫進度（0到1）。

        原理：
        - 計算動畫進度，公式：progress = min(timer / duration, 1.0)。
        - 用於渲染動畫效果，例如逐漸淡出或顯示死亡特效。

        Returns:
            float: 動畫進度，範圍 [0, 1]。
        """
        return min(self.death_animation_timer / self.death_animation_duration, 1.0)

    def end_game(self) -> None:
        """
        結束遊戲並儲存分數。

        原理：
        - 將遊戲運行狀態設為 False，停止遊戲主循環。
        - 調用 _save_game_data 儲存玩家分數和遊戲數據。
        """
        self.running = False
        self._save_game_data()

    def get_pacman(self) -> PacMan:
        """
        獲取 Pac-Man 物件。

        原理：
        - 返回當前的 PacMan 實例，供外部模塊（例如渲染或控制）訪問。

        Returns:
            PacMan: Pac-Man 實例。
        """
        return self.pacman

    def get_maze(self) -> Map:
        """
        獲取迷宮物件。

        原理：
        - 返回當前的 Map 實例，供渲染或路徑規劃使用。

        Returns:
            Map: 迷宮實例。
        """
        return self.maze

    def get_power_pellets(self) -> List[PowerPellet]:
        """
        獲取能量球列表。

        原理：
        - 返回當前的能量球列表，供遊戲邏輯或渲染使用。

        Returns:
            List[PowerPellet]: 能量球列表。
        """
        return self.power_pellets

    def get_score_pellets(self) -> List[ScorePellet]:
        """
        獲取分數球列表。

        原理：
        - 返回當前的分數球列表，供遊戲邏輯或渲染使用。

        Returns:
            List[ScorePellet]: 分數球列表。
        """
        return self.score_pellets

    def get_ghosts(self) -> List[Ghost]:
        """
        獲取鬼魂列表。

        原理：
        - 返回當前的鬼魂列表，供遊戲邏輯或渲染使用。

        Returns:
            List[Ghost]: 鬼魂列表。
        """
        return self.ghosts

    def get_lives(self) -> int:
        """
        獲取玩家剩餘生命數。

        原理：
        - 返回 Pac-Man 的當前生命數，供 UI 顯示或遊戲邏輯使用。

        Returns:
            int: 剩餘生命數。
        """
        return self.pacman.lives

    def get_final_score(self) -> int:
        """
        獲取最終分數。

        原理：
        - 調用 _save_game_data 儲存遊戲數據，然後返回 Pac-Man 的最終分數。
        - 用於遊戲結束時顯示分數或記錄排行榜。

        Returns:
            int: 最終分數。
        """
        self._save_game_data()
        return self.pacman.score

    def did_player_win(self) -> bool:
        """
        檢查玩家是否贏得遊戲。

        原理：
        - 若所有能量球和分數球都被吃完，則認為玩家勝利。

        Returns:
            bool: 是否贏得遊戲。
        """
        return not self.power_pellets and not self.score_pellets

    def _save_game_data(self) -> None:
        """
        儲存遊戲數據，包括玩家名稱、分數、迷宮種子和遊玩時長。

        原理：
        - 計算遊玩時長，公式：play_time = (end_time - start_time) / 1000。
        - 調用 menu 模塊的 save_score 函數，將玩家名稱、分數、迷宮種子和遊玩時長儲存。
        - 用於記錄玩家表現，生成排行榜或日誌。
        """
        from game.menu import save_score
        end_time = pygame.time.get_ticks()  # 獲取結束時間（毫秒）
        play_time = (end_time - self.start_time) / 1000.0  # 轉換為秒
        save_score(self.player_name, self.pacman.score, MAZE_SEED, play_time)