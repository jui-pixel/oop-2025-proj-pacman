# game/entities/entity_initializer.py
"""
提供遊戲實體的初始化功能。
"""
from pacman import PacMan
from ghost import Ghost1, Ghost2, Ghost3, Ghost4
from pellets import PowerPellet, ScorePellet
from typing import Tuple, List
import random

def initialize_entities(maze) -> Tuple[PacMan, List, List[PowerPellet], List[ScorePellet]]:
    """
    初始化所有遊戲實體，包括 Pac-Man、鬼魂、能量球和分數球。
    
    Args:
        maze: 迷宮物件。
    
    Returns:
        Tuple: (PacMan, List[Ghost], List[PowerPellet], List[ScorePellet])
    """
    # 尋找 Pac-Man 的起始位置
    valid_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                      if maze.get_tile(x, y) == '.']
    
    edge_mid_positions = []
    for x, y in valid_positions:
        is_near_edge = (x <= 2 or x >= maze.w - 3 or y <= 2 or y >= maze.h - 3)
        is_not_middle = not (maze.w // 4 < x < 3 * maze.w // 4 and maze.h // 4 < y < 3 * maze.h // 4)
        if is_near_edge and is_not_middle:
            edge_mid_positions.append((x, y))
    
    if edge_mid_positions:
        pacman_pos = random.choice(edge_mid_positions)
    else:
        if valid_positions:
            pacman_pos = random.choice(valid_positions)
        else:
            raise ValueError("No valid start position ('.') for Pac-Man in maze")
    
    pacman = PacMan(pacman_pos[0], pacman_pos[1])
    
    # 初始化鬼魂
    ghost_classes = [Ghost1, Ghost2, Ghost3, Ghost4]
    ghost_spawn_points = [(x, y) for y in range(maze.h) for x in range(maze.w) if maze.get_tile(x, y) == 'S']
    if not ghost_spawn_points:
        raise ValueError("迷宮中沒有'S'格子，無法生成鬼魂！")
    
    random.shuffle(ghost_spawn_points)
    ghosts = []
    for i, ghost_class in enumerate(ghost_classes):
        spawn_point = ghost_spawn_points[i % len(ghost_spawn_points)]
        ghosts.append(ghost_class(spawn_point[0], spawn_point[1], f"Ghost{i+1}"))
    
    # 初始化能量球
    power_pellets = []
    a_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                   if maze.get_tile(x, y) == 'E' and (x, y) != (pacman.x, pacman.y)]
    for x, y in a_positions:
        power_pellets.append(PowerPellet(x, y))
    
    # 初始化分數球
    all_path_positions = [(x, y) for y in range(1, maze.h - 1) for x in range(1, maze.w - 1)
                         if maze.get_tile(x, y) == '.']
    excluded_positions = [(pacman.x, pacman.y)] + ghost_spawn_points + a_positions
    score_positions = [pos for pos in all_path_positions if pos not in excluded_positions]
    score_pellets = [ScorePellet(x, y) for x, y in score_positions]
    
    return pacman, ghosts, power_pellets, score_pellets