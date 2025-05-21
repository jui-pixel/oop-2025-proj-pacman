# ghosts/ghost2.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan
from config import PINK

class Ghost2(BasicGhost):
    def __init__(self, x: int, y: int, name: str = "Ghost2"):
        super().__init__(x, y, name, color=PINK)

    def chase_pacman(self, pacman: PacMan, maze):
        """預測 Pac-Man 前進方向，瞄準前方 4 格。"""
        dx, dy = 0, 0
        if pacman.target_x > pacman.x:
            dx = 1
        elif pacman.target_x < pacman.x:
            dx = -1
        elif pacman.target_y > pacman.y:
            dy = 1
        elif pacman.target_y < pacman.y:
            dy = -1

        target_x = pacman.x + dx * 4
        target_y = pacman.y + dy * 4

        target_x = max(0, min(maze.w - 1, target_x))
        target_y = max(0, min(maze.h - 1, target_y))

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        min_distance = float('inf')

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'D', 'E']:
                distance = ((new_x - target_x) ** 2 + (new_y - target_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    best_direction = (dx, dy)

        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            return
        self.move_random(maze)