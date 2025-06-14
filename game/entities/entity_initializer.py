# game/entities/entity_initializer.py
"""
提供遊戲實體的初始化功能，負責創建 Pac-Man、鬼魂、能量球和分數球。

這個模組負責根據迷宮結構生成所有遊戲實體，確保它們的初始位置合理且不重疊。
"""

# 匯入相關的實體類
from .pacman import PacMan
from .ghost import Ghost1, Ghost2, Ghost3, Ghost4
from .pellets import PowerPellet, ScorePellet
# 匯入型別提示模組
from typing import Tuple, List
# 匯入隨機模組，用於隨機化位置
import random
# 從 config 檔案匯入迷宮瓦片類型
from config import TILE_PATH, TILE_POWER_PELLET, TILE_GHOST_SPAWN

def initialize_entities(maze) -> Tuple[PacMan, List, List[PowerPellet], List[ScorePellet]]:
    """
    初始化所有遊戲實體，包括 Pac-Man、鬼魂、能量球和分數球。

    原理：
    - 根據迷宮結構初始化遊戲實體，確保實體位置合理且不重疊。
    - Pac-Man 優先在靠近邊緣但非迷宮中心的有效路徑位置隨機生成，增加遊戲多樣性。
    - 鬼魂生成在指定的重生點（TILE_GHOST_SPAWN），隨機分配位置。
    - 能量球生成在迷宮的 TILE_POWER_PELLET 位置，分數球生成在所有有效路徑位置（排除 Pac-Man、鬼魂和能量球位置）。
    - 隨機化處理確保訓練數據的多樣性和遊戲的可玩性。

    Args:
        maze: 迷宮物件，提供瓦片信息和尺寸。

    Returns:
        Tuple: (pacman, ghosts, power_pellets, score_pellets)
        - pacman: PacMan 物件。
        - ghosts: 鬼魂列表（包含 Ghost1, Ghost2, Ghost3, Ghost4）。
        - power_pellets: 能量球列表。
        - score_pellets: 分數球列表。
    """
    # 尋找 Pac-Man 的起始位置
    # 收集迷宮中所有有效路徑格子（TILE_PATH），排除邊界
    valid_positions = [(x, y) for y in range(1, maze.height - 1) for x in range(1, maze.width - 1)
                      if maze.get_tile(x, y) == TILE_PATH]
    
    # 優先選擇靠近邊緣且非中心的有效位置
    edge_mid_positions = []
    for x, y in valid_positions:
        # 檢查是否靠近邊緣（距離邊界 2 格以內）
        is_near_edge = (x <= 2 or x >= maze.width - 3 or y <= 2 or y >= maze.height - 3)
        # 檢查是否遠離迷宮中心（不在迷宮寬高 1/4 到 3/4 的範圍內）
        is_not_middle = not (maze.width // 4 < x < 3 * maze.width // 4 and maze.height // 4 < y < 3 * maze.height // 4)
        if is_near_edge and is_not_middle:
            edge_mid_positions.append((x, y))
    
    # 如果有符合條件的邊緣位置，隨機選擇一個
    if edge_mid_positions:
        pacman_pos = random.choice(edge_mid_positions)
    else:
        # 否則從所有有效位置中隨機選擇
        if valid_positions:
            pacman_pos = random.choice(valid_positions)
        else:
            # 如果迷宮沒有有效路徑，拋出錯誤
            raise ValueError("迷宮中沒有有效路徑（'.'）用於生成 Pac-Man")
    
    # 初始化 Pac-Man 物件
    pacman = PacMan(pacman_pos[0], pacman_pos[1])
    
    # 初始化鬼魂
    # 定義四種鬼魂類型
    ghost_classes = [Ghost1, Ghost2, Ghost3, Ghost4]
    # 收集迷宮中所有鬼魂重生點（TILE_GHOST_SPAWN）
    ghost_spawn_points = [(x, y) for y in range(maze.height) for x in range(maze.width) if maze.get_tile(x, y) == TILE_GHOST_SPAWN]
    if not ghost_spawn_points:
        # 如果沒有重生點，拋出錯誤
        raise ValueError("迷宮中沒有 'S' 格子，無法生成鬼魂！")
    
    # 隨機打亂鬼魂出生點，增加多樣性
    random.shuffle(ghost_spawn_points)
    ghosts = []
    # 為每種鬼魂分配一個出生點
    for i, ghost_class in enumerate(ghost_classes):
        # 使用模運算循環分配出生點（如果重生點少於鬼魂數）
        spawn_point = ghost_spawn_points[i % len(ghost_spawn_points)]
        # 創建鬼魂物件並添加到列表
        ghosts.append(ghost_class(spawn_point[0], spawn_point[1], f"Ghost{i+1}"))
    
    # 初始化能量球
    power_pellets = []
    # 收集迷宮中所有能量球位置（TILE_POWER_PELLET），排除 Pac-Man 位置
    a_positions = [(x, y) for y in range(1, maze.height - 1) for x in range(1, maze.width - 1)
                   if maze.get_tile(x, y) == TILE_POWER_PELLET and (x, y) != (pacman.x, pacman.y)]
    # 在每個能量球位置創建 PowerPellet 物件
    for x, y in a_positions:
        power_pellets.append(PowerPellet(x, y))
    
    # 初始化分數球
    # 收集所有有效路徑位置（TILE_PATH）
    all_path_positions = [(x, y) for y in range(1, maze.height - 1) for x in range(1, maze.width - 1)
                         if maze.get_tile(x, y) == TILE_PATH]
    # 排除 Pac-Man、鬼魂重生點和能量球位置
    excluded_positions = [(pacman.x, pacman.y)] + ghost_spawn_points + a_positions
    # 計算剩餘可放置分數球的位置
    score_positions = [pos for pos in all_path_positions if pos not in excluded_positions]
    # 在每個剩餘位置創建 ScorePellet 物件
    score_pellets = [ScorePellet(x, y) for x, y in score_positions]
    
    # 返回所有初始化的實體
    return pacman, ghosts, power_pellets, score_pellets