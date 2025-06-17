# ai/environment.py
# 導入必要的 Python 模組
import os # 用於處理檔案路徑
import sys # 用於修改 Python 的模組搜尋路徑
# 將當前檔案的父目錄加入 Python 的搜尋路徑，這樣可以匯入其他資料夾的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np # 用於數值計算，尤其是陣列操作
from gym.spaces import Discrete, Box # 從 gym 模組導入動作空間和觀察空間的類型
from game.game import Game # 匯入遊戲核心邏輯的 Game 類
# 從 config 檔案匯入遊戲的常數設定，例如迷宮大小、隨機種子等
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, CELL_SIZE, FPS, EDIBLE_DURATION, GHOST_SCORES, TILE_PATH, TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN
import random # 用於隨機選擇動作或位置
from typing import Callable # 用於型別提示，指定函數型別
from game.maze_generator import Map # 匯入迷宮生成器

class PacManEnv(Game):
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED, ghost_penalty_weight=3.0):
        """
        初始化 Pac-Man 環境，提供強化學習的接口。

        這個類繼承自 Game 類，用來模擬 Pac-Man 遊戲，並提供強化學習需要的環境接口，例如狀態、動作、獎勵等。

        Args:
            width (int): 迷宮的寬度（格子數）。
            height (int): 迷宮的高度（格子數）。
            seed (int): 隨機種子，用來確保遊戲的隨機性可重現。
            ghost_penalty_weight (float): 鬼魂距離的懲罰權重，用來調整獎勵計算。
        """
        # 呼叫父類 Game 的初始化函數，設定玩家名稱為 "RL_Agent"，固定有 4 隻鬼魂
        super().__init__(player_name="RL_Agent")
        # 儲存迷宮的寬度和高度
        self.width = width
        self.height = height
        # 每個格子的大小（像素單位）
        self.cell_size = CELL_SIZE
        # 儲存隨機種子
        self.seed = seed
        # 鬼魂懲罰權重，用於計算與鬼魂距離相關的獎勵
        self.ghost_penalty_weight = ghost_penalty_weight
        # 計算總共有多少顆豆子（分數豆 + 能量豆）
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        # 已吃掉的豆子數量
        self.eaten_pellets = 0
        # 遊戲是否結束
        self.game_over = False
        # 當前分數
        self.current_score = 0
        # 上一步的分數，用於計算獎勵
        self.old_score = 0
        # 目前的幀數（用於控制遊戲更新頻率）
        self.frame_count = 0
        # 鬼魂移動的計數器，控制鬼魂移動頻率
        self.ghost_move_counter = 2
        # 狀態通道數，遊戲狀態用 6 個通道表示（詳見 _get_state）
        self.state_channels = 6
        # 狀態的形狀，是一個三維陣列 (通道數, 高度, 寬度)
        self.state_shape = (self.state_channels, self.height, self.width)
        # 動作空間，定義為 4 個離散動作（上、下、左、右）
        self.action_space = Discrete(4)
        # 觀察空間，定義為一個三維陣列，值範圍在 [0, 1]，資料型別為浮點數
        self.observation_space = Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)
        # 用於儲存上一次的形狀獎勵（用於計算獎勵變化）
        self.last_shape = None
        # 用於儲存上一次的移動方向（用於計算獎勵變化）
        self.last_action = None
        # 初始化已訪問格子集合
        self.visited_positions = set()
        self.visited_positions.add((self.pacman.x, self.pacman.y))  # 添加初始位置
        self.consecutive_wall_collisions = 0
        # 設定隨機種子，確保結果可重現
        np.random.seed(seed)
        # 印出初始化資訊，方便除錯
        print(f"初始化 PacManEnv：寬度={width}，高度={height}，種子={seed}，鬼魂數=4")

    def _get_state(self):
        """
        獲取遊戲的當前狀態，返回一個 6 通道的張量（三維陣列）。

        每個通道表示遊戲中的不同元素：
        - 通道 0：Pac-Man 的位置
        - 通道 1：能量豆的位置
        - 通道 2：分數豆的位置
        - 通道 3：可食用鬼魂的位置
        - 通道 4：普通鬼魂的位置
        - 通道 5：牆壁的位置

        Returns:
            np.ndarray: 形狀為 (6, height, width) 的浮點數陣列。
        """
        # 初始化一個全零的三維陣列，用來儲存狀態
        state = np.zeros(self.state_shape, dtype=np.float32)
        # 設置牆壁的位置
        for y in range(self.height):
            for x in range(self.width):
                # 如果該格子是邊界或牆壁，則在第 5 通道設為 1
                if self.maze.get_tile(x, y) in [TILE_BOUNDARY,TILE_WALL]:
                    state[5, y, x] = 1.0
        # 設置 Pac-Man 的位置，在第 0 通道設為 1
        state[0, self.pacman.target_y, self.pacman.target_x] = 1.0
        # 設置能量豆的位置，在第 1 通道設為 1
        for pellet in self.power_pellets:
            state[1, pellet.y, pellet.x] = 1.0
        # 設置分數豆的位置，在第 2 通道設為 1
        for pellet in self.score_pellets:
            state[2, pellet.y, pellet.x] = 1.0
        # 設置鬼魂的位置
        for ghost in self.ghosts:
            # 如果鬼魂正在返回出生點或等待中，則跳過
            if ghost.returning_to_spawn or ghost.waiting:
                continue
            # 如果鬼魂是可食用的，則在第 3 通道設為 1
            elif ghost.edible:
                state[3, ghost.target_y, ghost.target_x] = 1.0
            # 否則是普通鬼魂，則在第 4 通道設為 1
            else:
                state[4, ghost.target_y, ghost.target_x] = 1.0
                state[4, ghost.y, ghost.x] = 1.0
        return state

    def get_expert_action(self):
        """
        使用規則基礎的 AI 提供專家動作，作為強化學習的參考。

        這個方法模擬一個簡單的 AI 策略，讓 Pac-Man 根據遊戲規則移動，並返回對應的動作編號。

        Returns:
            int: 動作編號（0=上, 1=下, 2=左, 3=右）。
        """
        # 定義四個方向的移動向量（x, y））
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # 呼叫 Pac-Man 的規則移動方法，根據迷宮、豆子和鬼魂位置決定移動
        # print(f"({self.pacman.x}, {self.pacman.y})")
        success = self.pacman.rule_based_ai_move(self.maze, self.power_pellets, self.score_pellets, self.ghosts)
        # print(f"({self.pacman.x}, {self.pacman.y}) to ({self.pacman.target_x}, {self.pacman.target_y})")
        # 如果移動成功且有上一次的方向，則轉換為動作編號
        if success and self.pacman.last_direction:
            dx, dy = self.pacman.last_direction
            # print(f"({dx}, {dy})")
            # print(f"({dx}, {dy}), {self.pacman.lives}")
            for i, (dx_dir, dy_dir) in enumerate(directions):
                if dx == dx_dir and dy == dy_dir:
                    # print(i)
                    # print()
                    return i
        # 找出所有安全的移動方向（不會撞牆或進入禁止區域）
        safe_directions = [i for i, (dx, dy) in enumerate(directions) 
                          if self.maze.xy_valid(self.pacman.x + dx, self.pacman.y + dy)
                          and self.maze.get_tile(self.pacman.x + dx, self.pacman.y + dy) not in [TILE_BOUNDARY, TILE_WALL, TILE_DOOR, TILE_GHOST_SPAWN]]
        # 如果有安全方向，隨機選擇一個；否則返回默認動作 0
        return random.choice(safe_directions) if safe_directions else 0

    def reset(self, seed=MAZE_SEED, random_spawn_seed=0, random_maze_seed = 0):
        """
        重置遊戲環境，重新開始遊戲。

        這個方法會重新初始化迷宮、Pac-Man、鬼魂和豆子，並返回初始狀態。

        Args:
            seed (int): 隨機種子。
            random_spawn_seed (int): 用於隨機生成 Pac-Man 出生點的種子。
            random_maze_seed (int): 用於隨機生成迷宮的種子。

        Returns:
            tuple: (初始狀態, 空字典資訊)。
        """
        # 如果提供了種子，則設定隨機種子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        # 呼叫父類的初始化函數，重新設置遊戲
        super().__init__(player_name="RL_Agent")
        # 如果提供了隨機迷宮種子，則生成新迷宮
        if random_maze_seed != 0:
            self.maze = Map(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=self.seed+random_maze_seed)
            self.maze.generate_maze()
            # 重新初始化遊戲實體（Pac-Man、鬼魂、豆子）
            self.pacman, self.ghosts, self.power_pellets, self.score_pellets = self._initialize_entities()
            # 找出鬼魂的出生點
            self.respawn_points = [(x, y) for y in range(self.maze.height) for x in range(self.maze.width) 
                               if self.maze.get_tile(x, y) == TILE_GHOST_SPAWN]
        # 如果提供了隨機出生點種子，則隨機選擇 Pac-Man 的起始位置
        if random_spawn_seed != 0:
            random.seed(self.seed + random_spawn_seed)
            # 找出所有有效的路徑格子
            valid_positions = [(x, y) for y in range(1, self.maze.height - 1) for x in range(1, self.maze.width - 1)
                              if self.maze.get_tile(x, y) == TILE_PATH]
            # 隨機選擇一個位置作為 Pac-Man 的起始位置
            self.pacman.x, self.pacman.y = random.choice(valid_positions)
            self.pacman.initial_x = self.pacman.x
            self.pacman.initial_y = self.pacman.y
        # 計算總豆子數量
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        # 重置已吃豆子數量
        self.eaten_pellets = 0
        # 重置遊戲結束狀態
        self.game_over = False
        # 重置分數
        self.current_score = 0
        self.old_score = 0
        # 重置幀數
        self.frame_count = 0
        # 重置鬼魂移動計數器
        self.ghost_move_counter = 2
        # 用於儲存上一次的形狀獎勵（用於計算獎勵變化）
        self.last_shape = None
        # 用於儲存上一次的移動方向（用於計算獎勵變化）
        self.last_action = None
        # 初始化已訪問格子集合
        self.visited_positions = set()
        self.visited_positions.add((self.pacman.x, self.pacman.y))  # 添加初始位置
        self.consecutive_wall_collisions = 0
        # 獲取初始狀態
        state = self._get_state()
        # 返回初始狀態和空字典
        return np.array(state, dtype=np.float32), {}

    def update(self, fps: int, move_pacman: Callable[[], None]) -> None:
        """
        更新遊戲狀態，執行一次遊戲邏輯更新。

        Args:
            fps (int): 每秒幀數，控制遊戲速度
。
            move_pacman: Callable[[], int]): 一個函數 Pac-Man，用來移動函數來移動Pac-Man。

        Returns:
            None:
        """
        # 執行動作Pac-Man 移動
        move_pacman()
        # 檢查是否吃能量到豆子
        score_from_pellet = self.pacman.eat_pellet(self.power_pellets)
        # 如果吃到能量到豆子，設置所有鬼魂為可食用狀態
        if score_from_pellet > 0:
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)
        # 檢查是否吃到分數豆子
        score_from_pellet += self.pacman.eat_score_pellet(self.score_pellets)
        self.eaten_pellets += score_from_pellet // 2
        # 更新所有鬼魂的狀態
        for ghost in self.ghosts:
            # 如果鬼魂移動到了目標位置
            if ghost.move_towards_target(FPS):
                # 如果鬼魂正在返回出生點並到達出生點，設置為等待狀態
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == TILE_GHOST_SPAWN:
                    ghost.set_waiting(FPS)
                # 如果鬼魂仍在返回途中，繼續返回
                elif ghost.returning_to_spawn:
                    ghost.return_to_spawn(self.maze)
                # 否則按照正常邏輯移動鬼魂
                else:
                    # 每隔兩幀更新一次鬼魂位置
                    if self.frame_count % self.ghost_move_counter == 0:
                        # 更新鬼魂的實際像素坐標
                        ghost.current_x = ghost.target_x * CELL_SIZE + CELL_SIZE // 2
                        ghost.current_y = ghost.target_y * CELL_SIZE + CELL_SIZE // 2
                        # 更新鬼魂的格子坐標
                        ghost.x = ghost.target_x
                        ghost.y = ghost.target_y
                    else:
                        ghost.move(self.pacman, self.maze, FPS)
        # 檢查碰撞
        self._check_collision(FPS)
        # 如果所有豆子都被吃完，則遊戲勝利
        if not self.power_pellets and not self.score_pellets:
            print(f"遊戲勝利！最終分數：{self.pacman.score}")
            self.running = False
            self.game_over = True

    def _check_collision(self, fps: int) -> None:
        """
        檢查 Pac-Man 與鬼魂的碰撞。

        根據碰撞結果，更新遊戲狀態（例如得分或失去生命）。

        Args:
            fps (int): 每秒幀數。
        """
        # 如果遊戲已結束，則不執行檢查
        if not self.running:
            return
        # 檢查與每個鬼魂的距離
        for ghost in self.ghosts:
            # 計算 Pac-Man 與鬼魂的像素距離
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            # 如果距離小於半個格子大小，則發生碰撞
            if distance < CELL_SIZE / 2:
                # 如果鬼魂是可食用的，則得分並讓鬼魂返回出生點
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    ghost.set_returning_to_spawn(FPS)
                # 如果鬼魂是普通狀態，則 Pac-Man 扣一滴血
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    self.pacman.lose_life(self.maze)
                    # 讓所有鬼魂返回出生點
                    for g in self.ghosts:
                        g.set_returning_to_spawn(FPS)
                    # 如果 Pac-Man 沒血了，遊戲結束
                    if self.pacman.lives <= 0:
                        self.running = False
                        self.game_over = True
                        # 如果還有生命，中斷循環以重置
                    else:
                        break

    def step(self, action):
        """
        執行一個動作，並返回強化學習的五元組（s, a, t, r, i）。

        Args:
            action (int): 動作編號（0=上, 1=下, 2=左, 3=右）。

        Returns:
            tuple: (下一個狀態, 獎勵, 是否終結, 是否截斷, 資訊字典)。
        """
        # 檢查動作是否合法
        if not 0 <= action < 4:
            raise ValueError(f"無效動作：{action}")
        
        # 記錄當前位置
        prev_position = (self.pacman.x, self.pacman.y)
        
        # 標記是否成功移動和是否撞到牆
        moved = True
        wall_collision = False
        expert_action = self.get_expert_action()
        def move_pacman():
            nonlocal moved, wall_collision
            # 定義四個方向的移動向量
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            # 根據動作選擇移動方向
            dx, dy = directions[action]
            if not self.pacman.set_new_target(dx, dy, self.maze):
                wall_collision = True
                moved = False
            else:
                # 更新 Pac-Man 的像素坐標
                self.pacman.current_x = self.pacman.target_x * CELL_SIZE + CELL_SIZE // 2
                self.pacman.current_y = self.pacman.target_y * CELL_SIZE + CELL_SIZE // 2
                # 更新 Pac-Man 的格子坐標
                self.pacman.x = self.pacman.target_x
                self.pacman.y = self.pacman.target_y
                
        # 執行遊戲更新
        try:
            self.update(FPS, move_pacman)
        except Exception as e:
            raise RuntimeError(f"遊戲更新失敗：{str(e)}")
        
        # 如果成功移動，更新當前分數
        if moved:
            self.current_score = self.pacman.score
        # 計算基礎獎勵（分數變動的 10 倍）
        reward = (self.current_score - self.old_score) * 10
        # print(reward)
        # 更新舊分數
        self.old_score = self.current_score
        # 如果撞牆，給予負獎勵
        if wall_collision:
            self.consecutive_wall_collisions += 1
            reward -= 1 * (1 + 0.5 * self.consecutive_wall_collisions)
        else:
            self.consecutive_wall_collisions = 0
        # 如果更專家動作一樣
        if action == expert_action:
            reward += 1
        # 如果遊戲未結束且無正獎勵，給予時間懲罰
        if not self.game_over and reward <= 0:
            reward -= 10 * (1 - 0.5 * self.eaten_pellets/self.total_pellets)
        # 如果遊戲勝利，給予大額獎勵
        if not self.power_pellets and not self.score_pellets:
            reward += 500
        # 檢查是否進入新區域並給予獎勵
        current_position = (self.pacman.x, self.pacman.y)
        if moved and current_position != prev_position and current_position not in self.visited_positions:
            self.visited_positions.add(current_position)
            reward += 50  # 探索新區域的正向獎勵
        
        # 計算形勢獎勵（根據與鬼魂和豆子的距離）  
        shape = 0
        # 計算迷宮的最大距離（用於正規化）
        max_dist = (MAZE_WIDTH ** 2 + MAZE_HEIGHT ** 2) ** 0.5
        # 定義距離計算函數
        calc_dist = lambda y : max(1, ((self.pacman.x - y.x) ** 2 + (self.pacman.y - y.y) ** 2) ** 0.5)
        # 計算與鬼魂的距離懲罰
        for ghost in self.ghosts:
            dist = calc_dist(ghost)
            # 忽略正在返回或等待的鬼魂
            if ghost.returning_to_spawn or ghost.waiting:
                continue
            # 可食用鬼魂的距離獎勵
            elif ghost.edible:
                shape -= self.ghost_penalty_weight * (dist / max_dist) / max(1, len(self.ghosts)) * 0.5
            # 普通鬼魂的距離懲罰
            else:
                shape -= self.ghost_penalty_weight * (1 / max(1, dist)) / max(1, len(self.ghosts))
        # 計算與能量豆的距離獎勵
        for pellet in self.power_pellets:
            dist = calc_dist(pellet)
            shape -= self.ghost_penalty_weight * (dist / max_dist) / len(self.power_pellets) * 0.4
        # 計算與分數豆的距離獎勵
        for pellet in self.score_pellets:
            dist = calc_dist(pellet)
            shape -= self.ghost_penalty_weight * (dist / max_dist) / len(self.score_pellets) * 0.3
        
        
        reward += (shape * 0.95 - self.last_shape) if self.last_shape else 0
        # 儲存當前形勢獎勵
        self.last_shape = shape
        
        # 如果有上一不的移動方向，且與現在移動方向相反，且附近沒有鬼
        min_ghost_dist = min([calc_dist(ghost) for ghost in self.ghosts if not (ghost.returning_to_spawn or ghost.waiting)], default=max_dist)
        if self.last_action is not None and action == (self.last_action ^ 1) and min_ghost_dist > 3:
            reward -= 1
        self.last_action = action
        # 正規化獎勵
        reward = np.clip(reward, -500, 500)
        reward /= 500
        
        # 設定是否終結或截斷
        # truncated = self.game_over
        terminated = self.game_over
        # 獲取下一個狀態
        next_state = np.array(self._get_state(), dtype=np.float32)
        # 準備資訊字典
        info = {
            "frame_count": self.frame_count,
            "current_score": self.current_score,
            "eaten_pellets": self.eaten_pellets,
            "total_pellets": self.total_pellets,
            "valid_step": moved and not terminated,
            "lives_lost": self.pacman.lives < 3
        }
        # 增加幀數
        self.frame_count += 1
        # 返回四元組
        return next_state, reward, terminated, info

    def close(self):
        """
        關閉環境，清理資源。

        Returns:
            None:
        """
        # 呼叫父類的結束遊戲方法
        super().end_game()
        # 印出提示訊息
        print("環境已關閉")