# game/game.py
from typing import List, Tuple
from .entities import PacMan, Ghost, PowerPellet, ScorePellet
from .maze_generator import Map
from config import EDIBLE_DURATION, GHOST_SCORES, MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, PATH_DENSITY, FPS, CELL_SIZE

class Game:
    def __init__(self):
        """初始化遊戲。"""
        self.maze = Map(w=MAZE_WIDTH, h=MAZE_HEIGHT, seed=MAZE_SEED)
        self.maze.generate_connected_maze(path_density=PATH_DENSITY)
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = self._initialize_entities()
        self.respawn_points = [(x, y) for y in range(self.maze.h) for x in range(self.maze.w) if self.maze.get_tile(x, y) == 'S']
        self.ghost_score_index = 0
        self.running = True

    def _initialize_entities(self) -> Tuple[PacMan, List[Ghost], List[PowerPellet], List[ScorePellet]]:
        """初始化遊戲實體。"""
        from .entities import initialize_entities
        return initialize_entities(self.maze)

    def update(self, fps: int, move_pacman: callable) -> None:
        """更新遊戲狀態。

        Args:
            fps (int): 每秒幀數。
            move_pacman (callable): 移動 Pac-Man 的函數。
        """
        # 移動 Pac-Man
        move_pacman()

        # 檢查 Pac-Man 是否吃到能量球
        score_from_pellet = self.pacman.eat_pellet(self.power_pellets)
        if score_from_pellet > 0:
            for ghost in self.ghosts:
                ghost.set_edible(EDIBLE_DURATION)

        # 檢查 Pac-Man 是否吃到分數球
        self.pacman.eat_score_pellet(self.score_pellets)

        # 移動鬼魂
        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == 'S':
                    ghost.set_waiting(fps)
                else:
                    ghost.move(self.pacman, self.maze, fps)

        # 檢查碰撞
        self._check_collision(fps)

    def _check_collision(self, fps: int) -> None:
        """檢查 Pac-Man 與鬼魂的碰撞。

        Args:
            fps (int): 每秒幀數。
        """
        for ghost in self.ghosts:
            # 使用像素坐標進行碰撞檢測，放寬條件
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:  # 如果距離小於半個格子，認為發生碰撞
                if ghost.edible and ghost.edible_timer > 0:
                    self.pacman.score += GHOST_SCORES[self.ghost_score_index]
                    self.ghost_score_index = min(self.ghost_score_index + 1, len(GHOST_SCORES) - 1)
                    ghost.set_returning_to_spawn(fps)
                elif not ghost.edible and not ghost.returning_to_spawn and not ghost.waiting:
                    print(f"Game Over! Score: {self.pacman.score}")
                    self.running = False
                break
            