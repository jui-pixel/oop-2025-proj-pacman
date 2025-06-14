# game/game.py
"""
定義 Pac-Man 遊戲的核心邏輯，包括初始化、更新狀態和碰撞檢測。

這個模組負責管理遊戲的主要邏輯，例如迷宮生成、實體移動、碰撞處理和遊戲結束條件。
"""

# 匯入必要的模組
from typing import List, Tuple, Optional, Callable  # 用於型別提示，提升程式碼可讀性
from .entities.pacman import PacMan  # 匯入 Pac-Man 類，管理玩家角色
from .entities.ghost import *  # 匯入所有鬼魂類（Ghost, Ghost1, Ghost2 等）
from .entities.entity_initializer import initialize_entities  # 匯入實體初始化函數
from .entities.pellets import PowerPellet, ScorePellet  # 匯入能量球和分數球類
from .maze_generator import Map  # 匯入迷宮類，用於生成和管理迷宮
# 從 config 檔案匯入常數，例如遊戲設置、瓦片類型和分數
from config import EDIBLE_DURATION, GHOST_SCORES, MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, FPS, CELL_SIZE, TILE_GHOST_SPAWN
import config  # 匯入 config 模組，獲取全局設置
from collections import deque  # 用於可能的隊列操作（此處未直接使用）
import pygame  # 匯入 Pygame，用於時間管理和事件處理

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
        # 設置迷宮種子，確保迷宮生成一致
        self.seed = config.MAZE_SEED
        # 初始化迷宮物件，指定寬度和高度
        self.maze = Map(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=self.seed)
        # 生成隨機迷宮
        self.maze.generate_maze()
        # 初始化所有實體（Pac-Man、鬼魂、能量球、分數球）
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = self._initialize_entities()
        # 收集迷宮中所有鬼魂重生點的坐標（標記為 TILE_GHOST_SPAWN 的格子）
        self.respawn_points = [(x, y) for y in range(self.maze.height) for x in range(self.maze.width) 
                               if self.maze.get_tile(x, y) == TILE_GHOST_SPAWN]
        # 初始化鬼魂分數索引，用於追蹤連續吃鬼魂時的分數遞增
        self.ghost_score_index = 0
        # 設置遊戲運行狀態，初始為 True
        self.running = True
        # 記錄玩家名稱，用於儲存分數
        self.player_name = player_name
        # 記錄遊戲開始時間（毫秒）
        self.start_time = pygame.time.get_ticks()
        # 初始化死亡動畫狀態，初始為 False
        self.death_animation = False
        # 初始化死亡動畫計時器（幀數）
        self.death_animation_timer = 0
        # 設置死亡動畫持續時間（預設等於 FPS，相當於 1 秒）
        self.death_animation_duration = FPS

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
        # 調用外部模塊的 initialize_entities 函數，傳入迷宮物件
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
        # 如果正在播放死亡動畫
        if self.death_animation:
            # 增加動畫計時器
            self.death_animation_timer += 1
            # 當計時器超過動畫持續時間時，結束動畫
            if self.death_animation_timer >= self.death_animation_duration:
                self.death_animation = False
            return

        # 執行 Pac-Man 移動（由外部函數控制，例如玩家輸入或 AI）
        move_pacman()

        # 檢查是否吃到能量球，並獲取分數
        score_from_pellet = self.pacman.eat_pellet(self.power_pellets)
        if score_from_pellet > 0:
            # 如果吃到能量球，設置所有鬼魂為可食用狀態
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)

        # 檢查是否吃到分數球
        self.pacman.eat_score_pellet(self.score_pellets)

        # 移動所有鬼魂
        for ghost in self.ghosts:
            # 如果鬼魂到達當前目標格子
            if ghost.move_towards_target(FPS):
                # 如果鬼魂正在返回重生點且當前位置是重生點
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == TILE_GHOST_SPAWN:
                    ghost.set_waiting(fps)  # 進入等待狀態
                # 如果鬼魂正在返回重生點但尚未到達
                elif ghost.returning_to_spawn:
                    ghost.return_to_spawn(self.maze)  # 繼續返回重生點
                # 正常情況下執行鬼魂移動邏輯（追逐或逃跑）
                else:
                    ghost.move(self.pacman, self.maze, fps)

        # 檢查 Pac-Man 與鬼魂的碰撞
        self._check_collision(fps)

        # 檢查遊戲勝利條件：所有能量球和分數球都被吃完
        if not self.power_pellets and not self.score_pellets:
            print(f"遊戲勝利！所有彈丸已收集。最終分數：{self.pacman.score}")
            self.running = False  # 結束遊戲

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
        # 如果遊戲已結束，跳過碰撞檢查
        if not self.running:
            return
        
        # 檢查每個鬼魂
        for ghost in self.ghosts:
            # 計算 Pac-Man 與鬼魂的歐幾里得距離
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            # 如果距離小於半個格子尺寸，認為發生碰撞
            if distance < CELL_SIZE / 2:
                # 如果鬼魂可食用且可食用計時器未結束
                if ghost.edible and ghost.edible_timer > 0:
                    # 增加 Pac-Man 分數，根據當前鬼魂分數索引
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]
                    # 更新分數索引，確保不超過 GHOST_SCORES 長度
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    # 設置鬼魂返回重生點
                    ghost.set_returning_to_spawn(fps)
                # 如果鬼魂不可食用且不在返回或等待狀態
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    # Pac-Man 損失一條命，並重置到初始位置
                    self.pacman.lose_life(self.maze)
                    # 所有鬼魂返回重生點
                    for g in self.ghosts:
                        g.set_returning_to_spawn(fps)
                    # 如果 Pac-Man 生命數為 0
                    if self.pacman.lives <= 0:
                        self.running = False  # 結束遊戲
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
        from game.menu import save_score  # 延遲匯入，避免循環依賴
        # 獲取遊戲結束時間（毫秒）
        end_time = pygame.time.get_ticks()
        # 計算遊玩時長（秒）
        play_time = (end_time - self.start_time) / 1000.0
        # 儲存遊戲數據
        save_score(self.player_name, self.pacman.score, MAZE_SEED, play_time)