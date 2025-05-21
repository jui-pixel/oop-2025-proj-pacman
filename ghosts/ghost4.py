# ghosts/ghost4.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan
from config import LIGHT_BLUE
import random

class Ghost4(BasicGhost):
    def __init__(self, x: int, y: int, name: str = "Ghost4"):
        super().__init__(x, y, name, color = LIGHT_BLUE)

    def chase_pacman(self, pacman: PacMan, maze):
        """當靠近 Pac-Man 時隨機移動，遠離時追逐。"""
        distance = ((self.x - pacman.x) ** 2 + (self.y - pacman.y) ** 2) ** 0.5
        threshold = 8

        if distance < threshold:
            self.move_random(maze)
        else:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            best_direction = None
            min_distance = float('inf')

            for dx, dy in directions:
                new_x, new_y = self.x + dx, self.y + dy
                if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'D', 'E']:
                    distance = ((new_x - pacman.x) ** 2 + (new_y - pacman.y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_direction = (dx, dy)

            if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
                return
            self.move_random(maze)