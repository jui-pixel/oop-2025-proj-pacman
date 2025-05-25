# game/game.py
"""
定義 Pac-Man 遊戲的核心邏輯，包括初始化、更新狀態和碰撞檢測。
"""
from typing import List, Tuple
from .entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
from .maze_generator import Map
from config import EDIBLE_DURATION, GHOST_SCORES, MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, FPS, CELL_SIZE

class Game:
    def __init__(self):
        """
        初始化遊戲，設置迷宮、Pac-Man、鬼魂和其他實體。
        """
        self.maze = Map(w=MAZE_WIDTH, h=MAZE_HEIGHT, seed=MAZE_SEED)
        self.maze.generate_maze()  # 生成隨機迷宮
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = self._initialize_entities()
        self.respawn_points = [(x, y) for y in range(self.maze.h) for x in range(self.maze.w) 
                               if self.maze.get_tile(x, y) == 'S']  # 鬼魂重生點
        self.ghost_score_index = 0  # 鬼魂分數索引
        self.running = True  # 遊戲運行狀態

    def _initialize_entities(self) -> Tuple[PacMan, List[Ghost], List[PowerPellet], List[ScorePellet]]:
        """
        初始化遊戲中的所有實體（Pac-Man、鬼魂、能量球、分數球）。
        
        Returns:
            Tuple containing PacMan, list of Ghosts, list of PowerPellets, and list of ScorePellets.
        """
        return initialize_entities(self.maze)

    def update(self, fps: int, move_pacman: callable) -> None:
        """
        更新遊戲狀態，包括移動 Pac-Man、鬼魂和檢查碰撞。
        
        Args:
            fps (int): 每秒幀數，用於計時。
            move_pacman (callable): 控制 Pac-Man 移動的函數。
        """
        move_pacman()  # 移動 Pac-Man
        
        # 檢查是否吃到能量球
        score_from_pellet = self.pacman.eat_pellet(self.power_pellets)
        if score_from_pellet > 0:
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)  # 設置鬼魂為可吃狀態
        
        # 檢查是否吃到分數球
        self.pacman.eat_score_pellet(self.score_pellets)
        
        # 移動所有鬼魂
        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == 'S':
                    ghost.set_waiting(fps)  # 鬼魂到達重生點後等待
                else:
                    ghost.move(self.pacman, self.maze, fps)  # 執行鬼魂移動邏輯
        
        # 檢查碰撞
        self._check_collision(fps)
        
        if not self.power_pellets and not self.score_pellets:
            print(f"Game Won! All pellets collected. Final Score: {self.pacman.score}")
            self.running = False  # 遊戲結束

    def _check_collision(self, fps: int) -> None:
        """
        檢查 Pac-Man 與鬼魂的碰撞，根據鬼魂狀態更新分數或結束遊戲。
        
        Args:
            fps (int): 每秒幀數，用於計時。
        """
        for ghost in self.ghosts:
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:  # 碰撞檢測
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]  # 增加分數
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    ghost.set_returning_to_spawn(fps)  # 鬼魂返回重生點
                    ghost.return_to_spawn(maze=self.maze)
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    print(f"Game Over! Score: {self.pacman.score}")
                    self.running = False  # 遊戲結束
                break