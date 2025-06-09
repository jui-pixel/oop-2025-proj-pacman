# game/entities/entity_initializer.py
"""
提供遊戲實體的初始化功能，負責創建 Pac-Man、鬼魂、能量球和分數球。
"""

from .pacman import PacMan
from .ghost import Ghost1, Ghost2, Ghost3, Ghost4
from .pellets import PowerPellet, ScorePellet
from typing import Tuple, List
import random
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
    valid_positions = [(x, y) for y in range(1, maze.height - 1) for x in range(1, maze.width - 1)
                      if maze.get_tile(x, y) == TILE_PATH]
    
    # 優先選擇靠近邊緣且非中心的有效位置
    edge_mid_positions = []
    for x, y in valid_positions:
        is_near_edge = (x <= 2 or x >= maze.width - 3 or y <= 2 or y >= maze.height - 3)
        is_not_middle = not (maze.width // 4 < x < 3 * maze.width // 4 and maze.height // 4 < y < 3 * maze.height // 4)
        if is_near_edge and is_not_middle:
            edge_mid_positions.append((x, y))
    
    if edge_mid_positions:
        pacman_pos = random.choice(edge_mid_positions)  # 優先選擇邊緣位置
    else:
        if valid_positions:
            pacman_pos = random.choice(valid_positions)  # 回退到隨機有效位置
        else:
            raise ValueError("迷宮中沒有有效路徑（'.'）用於生成 Pac-Man")
    
    pacman = PacMan(pacman_pos[0], pacman_pos[1])  # 初始化 Pac-Man
    
    # 初始化鬼魂
    ghost_classes = [Ghost1, Ghost2, Ghost3, Ghost4]  # 四種鬼魂類型
    ghost_spawn_points = [(x, y) for y in range(maze.height) for x in range(maze.width) if maze.get_tile(x, y) == TILE_GHOST_SPAWN]
    if not ghost_spawn_points:
        raise ValueError("迷宮中沒有 'S' 格子，無法生成鬼魂！")
    
    random.shuffle(ghost_spawn_points)  # 隨機分配鬼魂出生點
    ghosts = []
    for i, ghost_class in enumerate(ghost_classes):
        spawn_point = ghost_spawn_points[i % len(ghost_spawn_points)]  # 循環分配出生點
        ghosts.append(ghost_class(spawn_point[0], spawn_point[1], f"Ghost{i+1}"))
    
    # 初始化能量球
    power_pellets = []
    a_positions = [(x, y) for y in range(1, maze.height - 1) for x in range(1, maze.width - 1)
                   if maze.get_tile(x, y) == TILE_POWER_PELLET and (x, y) != (pacman.x, pacman.y)]
    for x, y in a_positions:
        power_pellets.append(PowerPellet(x, y))  # 在能量球位置生成 PowerPellet
    
    # 初始化分數球
    all_path_positions = [(x, y) for y in range(1, maze.height - 1) for x in range(1, maze.width - 1)
                         if maze.get_tile(x, y) == TILE_PATH]
    excluded_positions = [(pacman.x, pacman.y)] + ghost_spawn_points + a_positions  # 排除 Pac-Man、鬼魂和能量球位置
    score_positions = [pos for pos in all_path_positions if pos not in excluded_positions]
    score_pellets = [ScorePellet(x, y) for x, y in score_positions]  # 在剩餘路徑生成分數球
    
    return pacman, ghosts, power_pellets, score_pellets