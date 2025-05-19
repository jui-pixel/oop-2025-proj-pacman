# ghosts/basic_ghost.py
from game.entities import Ghost, PacMan
from config import RED

class BasicGhost(Ghost):
    def __init__(self, x: int, y: int, name: str = "BasicGhost", color: tuple[int, int, int] = RED):
        super().__init__(x, y, name, color=color)

    def chase_pacman(self, pacman: PacMan, maze):
        """直接追逐 Pac-Man"""
        dx = pacman.x - self.x
        dy = pacman.y - self.y
        directions = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
        for dir_x, dir_y in directions:
            if self.set_new_target(dir_x, dir_y, maze):
                return
        self.move_random(maze)  # 如果無法追逐，則隨機移動